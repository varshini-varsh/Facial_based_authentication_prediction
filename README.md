# Facial_based_authentication_prediction

# Gender Prediction
Gender Prediction is a classification problem. The output layer in the gender prediction network, is of type softmax with 2 nodes indicating the two classes “Male” and “Female”.

# Age Prediction
Ideally, Age Prediction should be approached as a Regression problem since we are expecting a real number as the output. However, estimating age accurately using regression is challenging. Even humans cannot accurately predict the age based on looking at a person. However, we have an idea of whether they are in their 20s or in their 30s. Because of this reason, it is wise to frame this problem as a classification problem where we try to estimate the age group the person is in. For example, age in the range of 0-2 is a single class, 4-6 is another class and so on.

The Adience dataset has 8 classes divided into the following age groups [(0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100)]. Thus, the age prediction network has 8 nodes in the final softmax layer indicating the mentioned age ranges.

It should be kept in mind that Age prediction from a single image is not a very easy problem to solve as the perceived age depends on a lot of factors and people of the same age may look pretty different in various parts of the world.

## The code can be divided into four parts:
* Detect Faces
* Detect Gender
* Detect Age
* Display output

# Importing the libraries
* OpenCV
* Matplotlib # to plot image in jupyter notebook 

# Detect Face
Use the DNN Face Detector for face detection.The face detection is done using the function "highlightFace".

# Predict Gender
Load the gender network into memory and pass the detected face through the network. The forward pass gives the probabilities or confidence of the two classes and the max of two outputs will be used as the final gender prediction.

# Predict Age
Load the age network and use the forward pass to get the output. Since the network architecture is similar to the Gender Network, take the max out of all the outputs to get the predicted age group.

# Display Output
Display the output of the network on the input images and show them using the imshow function or plot using matplotlib


