#!/usr/bin/env python

"""
Create a list of files and sort them by IOV.
Loop on the list and fill the dictionary of values.
"""

import os
import time
import calendar

totalTrackerModules = 15148

outputFile = open("full.js", "w")

outputFile.write("{\n")
outputFile.write("    label: 'High Voltage ON',\n")
outputFile.write("    data: [")

dirList = os.listdir("./")

# print dirList

fileList = []

for inputFileName in dirList:
    if inputFileName.startswith("DetVOffReaderSummary") and inputFileName.endswith(".log"):
        firstDateString = inputFileName.split("_TO_")[0].split("_FROM_")[1]
        dateArray = firstDateString.replace("__", "_").split("_")
        firstTimeValue = calendar.timegm(time.strptime(dateArray[0]+" "+dateArray[1]+" "+dateArray[2]+" "+dateArray[3]+":"+dateArray[4]+":"+dateArray[5]+" "+dateArray[6]))
        firstTimeValue = firstTimeValue*1000
        
        dateString = inputFileName.split("_TO_")[1].split(".")[0]
        dateArray = dateString.replace("__", "_").split("_")
        lastTimeValue = calendar.timegm(time.strptime(dateArray[0]+" "+dateArray[1]+" "+dateArray[2]+" "+dateArray[3]+":"+dateArray[4]+":"+dateArray[5]+" "+dateArray[6]))
        lastTimeValue = lastTimeValue*1000
        
        fileList.append((firstTimeValue, lastTimeValue, inputFileName))

# Loop on the file list sorted with the time of the start of the IOV
first = True
#firstTimeValue = 0
#for inputFileName in dirList:
#    if inputFileName.endswith(".log"):
for fileBlock in sorted(fileList, key=lambda fileTuple: fileTuple[0]):
    inputFile = open(fileBlock[2], "r")
    totHVoff = 0
    totLVoff = 0
    counter = 0
    checkLine = False
    for line in inputFile:
        if "TIB" in line:
            checkLine = True
        if "Summary" in line and checkLine:
            totHVoff = counter
            counter = 0
            checkLine = False
        if "%MSG" in line and checkLine:
            totLVoff = counter
            counter = 0
            checkLine = False
        if checkLine and len(line) != 1:
            counter += int(line.rsplit(" ", 1)[1])
    
    # print "Total modules with HV off =", totHVoff
    # print "Total modules with LV off =", totLVoff
    if first:
        first = False
    else:
        outputFile.write(", ")
    outputFile.write("["+str(fileBlock[0])+", "+str(totalTrackerModules - totHVoff)+"], ")
    outputFile.write("["+str(fileBlock[1])+", "+str(totalTrackerModules - totHVoff)+"]")

#    if first:
#        print "firstTimeValue =", firstTimeValue
#        valuesDict[firstTimeValue] = totHVoff
#        first = False
#    print "timeValue =", timeValue
#    valuesDict[timeValue] = totHVoff
    # outputFile.write(str(totLVoff))

#    data: [[0, 30], [1, 39], [2, 20], [3, 12], [4, 5], [6, 6], [7, 20], [8, 31], [9, 29], [10, 9]]
# outputFile.write("]\n}")

# print sorted(valuesDict)

#first = True
#for key in sorted(valuesDict):
#    if not first:
#        outputFile.write(" ,")
#    else:
#        # outputFile.write("["+str(firstTimeValue)+", "+str(totHVoff)+"], ")
#        # valuesDict[firstTimeValue] = totHVoff
#        first = False
#    # print "key =", key, "value =", valuesDict[key]
#    outputFile.write("["+str(key)+", "+str(valuesDict[key])+"]")
    
outputFile.write("]\n}")
