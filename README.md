# Windows 7 Processor Identification Model

Welcome to the Windows 7 Processor Identification Model project! This project involves building a Convolutional Neural Network (CNN) to identify the type of processor associated with Windows 7 based on certain features. The model utilizes TensorFlow and Keras for implementation.

## Overview
The primary goal of this project is to predict the type of processor for Windows 7 based on a given dataset. The dataset, assumed to be in the 'Train_Test_Windows_7.csv' file, is loaded and preprocessed to handle missing values and scale features. The CNN architecture is designed to capture relevant patterns for processor identification.

## Prerequisites
- Python 3.x
- Pandas
- NumPy
- TensorFlow
- Keras
- Scikit-Learn
- Matplotlib
- Seaborn

## How to Use
1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Install the required dependencies using pip:
   ```
   pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
   ```
3. **Prepare the Dataset**: Ensure your dataset is named 'Train_Test_Windows_7.csv' and is formatted appropriately with features and labels.
4. **Run the Code**: Execute the main script, adjusting the file path if necessary.
   ```python
   python windows7_processor_model.py
   ```
5. **Review Results**: The script will provide data analysis, preprocess the data, build and train the CNN model, and evaluate its performance. Evaluation metrics and visualizations will be displayed.

## CNN Model Architecture
The CNN model is designed with convolutional layers, max-pooling layers, and dense layers to effectively capture patterns in the data. Adjustments can be made to the model architecture based on specific requirements.

## Data Analysis and Preprocessing
The initial steps involve analyzing the dataset, checking for missing values, and preprocessing the data to handle any inconsistencies. The 'label' column is considered as the target variable.

## Evaluation Metrics and Visualizations
The model's performance is evaluated using standard metrics such as accuracy, precision, recall, and F1 score. Additionally, visualizations, including loss and accuracy plots, and a confusion matrix, provide insights into the model's behavior.

Feel free to customize and enhance the project based on your specific requirements!

## Contributors
- [Vijay Sai Kumar](https://github.com/vijay-svsk)
- [Sri Harsh](https://github.com/sriharsh-2003)
- [Kailash Varma](https://github.com/kailash123varma)

---
*Note: This project is designed for processor identification using a CNN model for Windows 7 datasets. Adjustments may be needed for different datasets or purposes.*
