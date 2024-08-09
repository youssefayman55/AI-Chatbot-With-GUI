import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
import pyttsx3
import threading

class ChatBot:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 155)  # Adjust speech rate (words per minute)

        # Predefined responses
        self.responses = {
            "hello" : "hello" ,
            "how are you": "I'm doing well, thank you for asking!",
            "what is your name": "I'm a ChatBot, nice to meet you!",
            "can you help me" : "yes, what do you want ?",
            "stop": "Goodbye! Take care.",
            "what is ai" : "AI is the simulation of human intelligence in machines that are programmed to think and learn like humans, performing tasks such as speech recognition, decision-making, and problem-solving" ,
            "what are the main types of ai" : "AI is classified into three main types: Narrow AI (ANI), General AI (AGI), and Superintelligent AI (ASI)." ,
            "what is machine learning" : "machine learning is a subset of AI that involves training algorithms to learn from and make predictions or decisions based on data." ,
            "what is deep learning" : "Deep Learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to analyze various factors of data." ,
            "what is supervised learning" : "Supervised learning is an machine learning technique where the model is trained on labeled data, meaning the input data is paired with the correct output." ,
            "what is unsupervised learning" : "Unsupervised learning is an machine learning technique where the model is trained on data without labels, and it tries to find hidden patterns or intrinsic structures in the input data." ,
            "what is reinforcement learning" : "Reinforcement learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to achieve maximum cumulative reward." ,
            "what is neural network" : "A neural network is a series of algorithms that attempt to recognize relationships in a set of data through a process that mimics the way the human brain operates." ,
            "what is natural language processing" : "natural language processing is a field of AI focused on the interaction between computers and humans through natural language, enabling machines to understand, interpret, and generate human language." ,
            "what is computer vision" : "Computer vision is a field of AI that enables machines to interpret and make decisions based on visual data from the world." ,
            "what is data analysis":"Data analysis is the process of inspecting, cleansing, transforming, and modeling data to discover useful information, draw conclusions, and support decision-making. It involves a variety of techniques and methods to extract meaningful insights from raw data.",
            "what is data science":"Data science is an interdisciplinary field that utilizes scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. It combines elements of statistics, mathematics, computer science, domain expertise, and communication skills to understand complex phenomena and solve real-world problems.",
            "what is data cleaning":" Identifying and correcting errors, inconsistencies, and missing values in the data to ensure its accuracy and reliability.",
            "what is data transformation":" Manipulating and restructuring data to make it suitable for analysis. This includes tasks like normalization, aggregation, and feature engineering.",
            "what is data visualisation":"Presenting data in visual formats such as charts, graphs, and maps to facilitate understanding and communication of insights.",
            "what is feature engineering":" Creating new features or variables from existing data that may improve the performance of machine learning models.",
            "what is chatbot" : "A chatbot is an AI-based software designed to simulate conversation with human users, especially over the internet." ,
            "what is the difference between AI and Machine Learning" : "AI is the broader concept of machines being able to carry out tasks in a smart way. ML is a subset of AI that involves the idea of systems learning from data." ,
            "what is data mining" : "Data mining is the process of discovering patterns and knowledge from large amounts of data, often using AI and Machine Learning techniques." ,
            "what is the difference between classification and regression in Machine learning" : "Classification is the process of predicting the class or category of a given data point, while regression predicts a continuous value." ,
            "what is overfitting in Machine learning" : "Overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor performance on new, unseen data." ,
            "what is underfitting in Machine learning" : "Underfitting happens when model is too simple to capture the underlying patterns in the data, leading to poor performance both on the training and test data." ,
            "what is svm algorithm" : "SVM is a supervised learning model used for classification and regression tasks that finds the hyperplane that best separates the data into different classes." ,
            "what is cnn algorithm" : "A CNN is a type of deep neural network commonly used for analyzing visual data, which uses convolutional layers to automatically and adaptively learn spatial hierarchies of features from input images." ,
            "what is decision tree algorithm" : "A decision tree is a decision support tool that uses a tree-like graph of decisions and their possible consequences, including chance event outcomes, resource costs, and utility" ,
            "what is random forest algorithm" : "A random forest is an ensemble learning method that constructs multiple decision trees during training and merges their results for better accuracy and robustness" ,
            "what is logistic regression algorithm" : "Logistic regression is a statistical model used for binary classification problems, predicting the probability of a binary outcome using a logistic function" ,
            "what is linear regression algorithm" : "Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more independent variables using a linear equation" ,
            "what is knn algorithm" : "KNN is a simple, instance-based learning algorithm that classifies new data points based on the majority class of their k-nearest neighbors in the training set" ,
            "what is naive bayes algorithm" : "Naive Bayes is a probabilistic classifier based on Bayes theorem, assuming independence between features, used for classification tasks" ,
            "what is rnn algorithm" : "RNN is a type of neural network designed for sequential data, where connections between nodes form a directed graph along a temporal sequence, allowing it to exhibit dynamic temporal behavior" ,
            "what is gradient boosting algorithm" : "Gradient boosting is an ensemble technique that builds models sequentially, each new model correcting the errors of the previous ones, typically used for regression and classification" ,
            "what is AdaBoost algorithm" : "AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple weak classifiers to create a strong classifier by focusing more on hard-to-classify instances" ,
            "what is XGBoost algorithm" : "XGBoost (Extreme Gradient Boosting) is an optimized gradient boosting library designed to be highly efficient , flexible , and portable, often used in data science competitions" ,
            "what is artificial neural network" : "artificial neural networks are the simplest form of neural networks consisting of an input layer, one or more hidden layers, and an output layer. Each layer  consists of nodes (neurons) connected by weights" ,
            "what is transformer Network" : "Transformers use self-attention mechanisms to process sequential data without relying on recurrence. They are widely used in natural language processing tasks like translation and text generation" ,
            "what is rule of big data in AI" : "Big data provides the vast amounts of information required to train complex AI models, enabling them to learn and make more accurate predictions" ,
            "what is back propagation" : "Backpropagation is an algorithm used in training neural networks, where the gradient of the loss function is calculated and used to update the weights in the network" ,
            "what is auto encoder" : "An autoencoder is a type of neural network used for unsupervised learning that aims to learn a compressed representation of the input data by encoding and then decoding it" ,
            "what is transformer model" : "A transformer model is a type of deep learning model that relies on self-attention mechanisms to process sequential data , widely used in NLP tasks" ,
            "what is boosting in machine learning" : "Boosting is an ensemble technique that combines multiple weak learners to create a strong learner by sequentially focusing on the errors of the previous models" ,
            "what is recommended system" : "A recommender system is an information filtering system that seeks to predict the rating or preference a user would give to an item, commonly used in online services" ,
            "what is anomaly detection" : "Anomaly detection is the identification of rare items, events, or observations that deviate significantly from the majority of the data, often used for fraud detection and monitoring" ,
            "what is wave net" : "WaveNet is a deep generative model for producing raw audio waveforms, capable of generating highly realistic speech and music" ,
        }

        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("ChatBot")

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=20, font=("Arial", 12))
        self.text_area.pack(padx=10, pady=10)

        self.entry = tk.Entry(self.root, font=("Arial", 12), width=40)
        self.entry.pack(padx=10, pady=5, side=tk.LEFT)
        self.entry.bind("<Return>", self.enter_pressed)

        self.send_button = tk.Button(self.root, text="Send", command=self.send_message)
        self.send_button.pack(padx=10, pady=5, side=tk.LEFT)

        self.voice_button = tk.Button(self.root, text="Voice Input", command=self.voice_input)
        self.voice_button.pack(padx=10, pady=5, side=tk.LEFT)

    def listen(self):
        with sr.Microphone() as source:
            self.update_text_area("Listening...")
            self.engine.say("Listening...")
            self.engine.runAndWait()
            self.recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
            audio = self.recognizer.listen(source)

        try:
            self.update_text_area("Processing audio...")
            self.engine.say("Processing audio...")
            self.engine.runAndWait()
            text = self.recognizer.recognize_google(audio)
            self.update_text_area(f"You: {text}")
            return text

        except sr.UnknownValueError:
            self.update_text_area("Sorry, I do not understand you.")
            return ""

        except sr.RequestError as e:
            self.update_text_area(f"Could not request results; {e}")
            return ""

    def respond(self, text):
        if not text:
            return

        # Check if the query has a predefined response
        response = self.responses.get(text.lower())
        if response:
            self.update_text_area(f"ChatBot: {response}")
            self.engine.say(response)
            self.engine.runAndWait()

        else:
            # Default response for unrecognized queries
            default_response = "Sorry, I'm not sure how to respond to that."
            self.update_text_area(f"ChatBot: {default_response}")
            self.engine.say(default_response)
            self.engine.runAndWait()

    def send_message(self):
        user_input = self.entry.get()
        self.update_text_area(f"You: {user_input}")
        self.respond(user_input)
        self.entry.delete(0, tk.END)

    def enter_pressed(self, event):
        self.send_message()

    def voice_input(self):
        threading.Thread(target=self.handle_voice_input).start()

    def handle_voice_input(self):
        user_input = self.listen()
        self.respond(user_input)
        if user_input.lower() == 'exit':
            self.root.quit()

    def update_text_area(self, text):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def start(self):
        self.engine.say("Hi , I'm your ChatBot. How can I help you?")
        self.engine.runAndWait()
        self.root.mainloop()

if __name__ == "__main__":
    bot = ChatBot()
    bot.start()



