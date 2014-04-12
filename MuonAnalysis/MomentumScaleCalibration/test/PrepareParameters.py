# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:38:26 2011

@author: Marco De Mattia, marco.de.mattia@cern.ch

This script goes with the template_MuScleFit_cfg.py. It reads the scale
parameters from the FitParameters.txt file (the first set it finds) and
it creates a MuScleFit_cfg.py from the template with them.
It does not replace the parFix and parOrder.
"""

functionNumber = 0
value = []
error = []

inputFile = open("FitParameters.txt",'r')

for line in inputFile:
    if "Fitting with resolution, scale, bgr function" in line:
        functionNumber = line.split("# ")[1].split(" ")[0]
    if "Results of the fit: parameter" in line:
        valueAndError = line.split("value")[1].split(" ")[1].split("+-")
        value.append(valueAndError[0])
        error.append(valueAndError[1])
        # print valueAndError[0], "+-", valueAndError[1]
        
    if "Scale" in line:
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

print values
    
cfgFile = open("template_MuScleFit_cfg.py", 'r')
outputCfgFile = open("MuScleFit_cfg.py", 'w')
for line in cfgFile:
    if "ResolFitType = " in line:
        outputCfgFile.write(line.replace("cms.int32(20)","cms.int32("+functionNumber+")"))
    elif "parResol = cms.vdouble()," in line:
        outputCfgFile.write(line.replace("parResol = cms.vdouble(),","parResol = cms.vdouble("+values+"),"))
    else:
        outputCfgFile.write( line )
    # print line
