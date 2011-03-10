#!/usr/bin/env python

"""
This macro allows to generate the background function to be put in Functions.h. The procedure is:
- run MuScleFit in bins of (eta_mu1, eta_mu2) determined by the "ranges" below
- fit the plots with CrystalBall+exponential for the J/Psi (one can change the macro to do different fits)
The second step requires RooFit. If RooFit is not available on the machine with CMSSW you can comment/uncomment
the regions as explained by the comments in the code to do only the first/second step.

Additional requirements for the macro are:
- the templated MuScleFit cfg: MuScleFit_cfg.py
- the templated fitting macro: CrystalBallFitOnData_JPsi.cc

To actually build the background function for MuScleFit, after all the steps in this macro are done you can
run BuildBackgroundFunction.py.
"""

import os

os.system("rm log")

templateFile = open("MuScleFit_cfg.py")
templateString = templateFile.read()
templateFile.close()

templateMacro = open("CrystalBallFitOnData_JPsi.cc")
templateMacroString = templateMacro.read()
templateMacro.close()

ranges = [
    ["0.", "0.85"],
    ["0.85", "1.25"],
    ["1.25", "1.45"],
    ["1.45", "1.55"],
    ["1.55", "1.65"],
    ["1.65", "1.75"],
    ["1.75", "1.85"],
    ["1.85", "1.95"],
    ["1.95", "1000."]
    ]

subRange = []
for cuts in ranges:
    subRange += [["-"+cuts[1], "-"+cuts[0], cuts[0], cuts[1], "True"]]

for i in range(len(ranges)):
    for j in range(i+1, len(ranges)):
        subRange += [[ranges[i][0], ranges[i][1], ranges[j][0], ranges[j][1], "False"]]

for cuts in subRange:

    print "cuts = ", cuts
    outputName = "MuScleFit_"+cuts[0]+"_"+cuts[1]+"_"+cuts[2]+"_"+cuts[3]
    outputRootFile = "0_"+outputName+".root"
    
    # # Uncomment this part if you are running on MuScleFit to produce the plots
    # # ------------------------------------------------------------------------
    # fullFile = templateString.replace("MIN_MUONETAFIRSTRANGE", cuts[0])
    # fullFile = fullFile.replace("MAX_MUONETAFIRSTRANGE", cuts[1])
    # fullFile = fullFile.replace("MIN_MUONETASECONDRANGE", cuts[2])
    # fullFile = fullFile.replace("MAX_MUONETASECONDRANGE", cuts[3])
    # fullFile = fullFile.replace("SEPARATERANGES", cuts[4])
    # fullFile = fullFile.replace("OUTPUTNAME", outputRootFile)
    # 
    # outputCfgName = outputName+"_cfg.py"
    # newFile = open(outputCfgName, 'w')
    # newFile.write(fullFile)
    # newFile.close()
    # 
    # print "Running", outputCfgName
    # os.system("cmsRun "+outputCfgName)
    # os.system("mv FitParameters.txt FitParameters"+outputName+".txt")


    # Uncomment this part if you are fitting the plots (requires RooFit)
    # ------------------------------------------------------------------
    print "Fitting", outputRootFile

    # outputMacroName = "macro_"+outputName
    outputMacroName = "macro"
    fullMacro = templateMacroString.replace("INPUTFILENAME", outputRootFile)
    fullMacro = fullMacro.replace("CrystalBallFitOnData_JPsi", outputMacroName)
    fullMacro = fullMacro.replace("FITRESULT", "macro_"+outputName+".root")
    newMacro = open(outputMacroName+".cc", 'w')
    newMacro.write(fullMacro)
    newMacro.close()
    
    os.system("echo "+outputName+" "+cuts[4]+" >> log")
    os.system("root -b -q -x -l "+outputMacroName+".cc | grep \"parameter =\" >> log")
    os.system("echo >> log")
