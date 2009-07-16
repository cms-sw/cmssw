#!/usr/bin/env python

""" This is script runs all the macros in the local Macros dir
"""

def arguments(comp, resType = "Z", firstFile = "0", secondFile = "1") :
    if( comp == "Pt" ) :
        name = "pt"
    elif( comp == "Eta" ) :
        name = "eta"
    elif( comp == "PhiPlus" ) :
        name = "phiPlus"
    elif( comp == "PhiMinus" ) :
        name = "phiMinus"
    else :
        print "Error"
        return ""
    if( resType == "Z" ) :
        fitType = "2"
    else :
        fitType = "1"
    return "\"hRecBestResVSMu_MassVS"+comp+"\", \""+firstFile+"_MuScleFit.root\", \""+secondFile+"_MuScleFit.root\", \"Resonance mass vs "+name+"\", \""+resType+"\", 4, 4, "+fitType+", \"filegraph_"+name+".root\""

import os

from ROOT import gROOT

firstFile = "\"0\""
secondFile = "\"3\""
resonanceType = "Z"
massProbablityName = "Z"

macrosDir = os.popen("echo $CMSSW_BASE", "r").read().strip()
macrosDir += "/src/MuonAnalysis/MomentumScaleCalibration/test/Macros/"

print macrosDir+"Run.C"

# Mass vs pt, eta, phi
# --------------------
gROOT.ProcessLine(".L "+macrosDir+"fit2DProj.C+");
fileNum1 = firstFile.strip("\"")
fileNum2 = secondFile.strip("\"")
gROOT.ProcessLine( "macroPlot("+arguments("Pt", resonanceType, fileNum1, fileNum2)+")" )
gROOT.ProcessLine( "macroPlot("+arguments("Eta", resonanceType, fileNum1, fileNum2)+")" )
gROOT.ProcessLine( "macroPlot("+arguments("PhiPlus", resonanceType, fileNum1, fileNum2)+")" )
gROOT.ProcessLine( "macroPlot("+arguments("PhiMinus", resonanceType, fileNum1, fileNum2)+")" )

# Resolution
# ----------
# The second parameter is a bool defining whether it should do half eta
# The third parameter is an integer defining the minimum number of entries required to perform a fit
gROOT.ProcessLine(".x "+macrosDir+"ResolDraw.cc+("+firstFile+", false, 100)")
gROOT.ProcessLine(".x "+macrosDir+"ResolDraw.cc+("+secondFile+", false, 100)")
gROOT.ProcessLine(".x "+macrosDir+"ResolCompare.cc("+firstFile+", "+secondFile+")")
# os.system("root -l "+macrosDir+"ResolCompare.cc")

# Pt reco vs Pt gen
gROOT.ProcessLine(".x "+macrosDir+"CompareRecoGenPt.C("+firstFile+", "+secondFile+")")

# Mass vs mass probability
gROOT.ProcessLine(".x "+macrosDir+"Plot_mass.C+("+firstFile+", "+secondFile+")")
gROOT.ProcessLine(".x "+macrosDir+"ShowMassComparison.C+(\""+massProbablityName+"\")")
# os.system("root -l "+macrosDir+"ShowMassComparison.C")
