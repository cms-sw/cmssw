# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:38:26 2011

@author: -
"""

functionNumber = 0
value = []
error = []

inputFile = open("FitParameters.txt",'r')

totalFits = 0
for line in inputFile:
    if "Fitting with resolution, scale, bgr function" in line:
        totalFits += 1
# print "totalFits = ", totalFits

inputFile = open("FitParameters.txt",'r')
actualFit = 0
for line in inputFile:
    if "Fitting with resolution, scale, bgr function" in line and actualFit == totalFits-1:
        functionNumber = line.split("# ")[1].split(" ")[0]
        # print "functionNumber =", functionNumber
        actualFit += 1
    elif "Fitting with resolution, scale, bgr function" in line:
        actualFit += 1
        # print "actualFit = ", actualFit
    if "Results of the fit: parameter" in line and actualFit == totalFits:
        valueAndError = line.split("value")[1].split(" ")[1].split("+-")
        value.append(valueAndError[0])
        error.append(valueAndError[1])
        # print "value = ", valueAndError
        # print valueAndError[0], "+-", valueAndError[1]
        
    if "Scale" in line and actualFit == totalFits:
        break

values = ""
errors = ""
errorParameters = ""
prepend = ""
for i in range(len(value)):
    values += prepend+str(value[i])
    errors += prepend+str(error[i])
    errorParameters += prepend+"1"
    prepend = ", "

print "values = ", values
print "errors = ", errors
    
cfgFile = open("ErrorsPropagationAnalyzer_cfg.py", 'r')
outputCfgFile = open("Errors_cfg.py", 'w')
for line in cfgFile:
    if "ResolFitType = " in line:
        outputCfgFile.write(line.replace("cms.int32(20)","cms.int32("+functionNumber+")"))
    elif "Parameters = cms.vdouble()," in line:
        outputCfgFile.write(line.replace("Parameters = cms.vdouble(),","Parameters = cms.vdouble("+values+"),"))
    elif "Errors = cms.vdouble()," in line:
        outputCfgFile.write(line.replace("Errors = cms.vdouble(),","Errors = cms.vdouble("+errors+"),"))
    elif "ErrorFactors = cms.vint32()," in line:
        outputCfgFile.write(line.replace("ErrorFactors = cms.vint32(),", "ErrorFactors = cms.vint32("+errorParameters+"),"))
    else:
        outputCfgFile.write( line )
    # print line
