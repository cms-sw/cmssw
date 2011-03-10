#!/usr/bin/env python

import os

os.system("rm fraction.txt")
os.system("rm backgroundShape.txt")

def functionBuilder(lofFile, line, outputFileName, addition, condition):
    fraction = logFile.readline()
    fractionAndError = fraction.split("=")[1].split("+-")
    fraction = fractionAndError[0].strip()
    fractionError = fractionAndError[1].strip()

    # fractionValue = str(float(fraction)+float(fractionError))
    # fractionValue = str(float(fraction)-float(fractionError))
    fractionValue = str(float(fraction))
    print fraction, "+-", fractionError
    conditionExp = "if"
    if condition:
        conditionExp = "else if"
    if line.find("True") != -1:
        os.system("echo \""+conditionExp+"( (fabs(eta1) >= "+cuts[3]+" && fabs(eta1) < "+cuts[4]+") && (fabs(eta2) >= "+cuts[3]+" && fabs(eta2) < "+cuts[4]+") ) {\" >> "+outputFileName)
    else:
        os.system("echo \""+conditionExp+"( ((fabs(eta1) >= "+cuts[1]+" && fabs(eta1) < "+cuts[2]+") && (fabs(eta2) >= "+cuts[3]+" && fabs(eta2) < "+cuts[4]+")) ||\" >> "+outputFileName)
        os.system("echo \"    ((fabs(eta2) >= "+cuts[1]+" && fabs(eta2) < "+cuts[2]+") && (fabs(eta1) >= "+cuts[3]+" && fabs(eta1) < "+cuts[4]+")) ) {\" >> "+outputFileName)
    if addition == "":
        os.system("echo \"  Bgrp2 = ("+fractionValue+");\" >> "+outputFileName)
    else:
        os.system("echo \"  return ("+addition+fractionValue+");\" >> "+outputFileName)
    os.system("echo \"}\" >> "+outputFileName)
    

print "Creating the C++ code"
logFile = open("log", 'r')
line = logFile.readline()
first = True
while line:
    print line
    if line.find("True") != -1 or line.find("False") != -1:
        cuts = line.split("_")
        cuts[4] = cuts[4].split(" ")[0]
        print "cuts =", cuts
        logFile.readline()
        logFile.readline()

        functionBuilder(logFile, line, "fraction.txt", "1.-", False)
        functionBuilder(logFile, line, "backgroundShape.txt", "", not first)

        if first == True:
            first = False

    line = logFile.readline()
