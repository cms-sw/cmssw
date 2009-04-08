#!/usr/bin/env python

""" This is script runs all the macros in the local Macros dir
"""

def arguments(comp, resType = "Z") :
    if( comp == "Pt" ) :
        name = "pt"
    elif( comp == "Eta" ) :
        name = "eta"
    elif( comp == "PhiPlus" ) :
        name = "phi"
    else :
        print "Error"
        return ""
    return "\"hRecBestResVSMu_MassVS"+comp+"\", \"0_MuScleFit.root\", \"3_MuScleFit.root\", \"Resonance mass vs "+name+"\", \""+resType+"\", 4, 4, 2, \"filegraph_"+name+".root\""

import os

from ROOT import gROOT

macrosDir = os.popen("echo $CMSSW_BASE", "r").read().strip()
macrosDir += "/src/MuonAnalysis/MomentumScaleCalibration/test/Macros/"

print macrosDir+"Run.C"

# Mass vs pt, eta, phi
# --------------------
gROOT.ProcessLine(".L "+macrosDir+"fit2DProj.C+");
gROOT.ProcessLine( "macroPlot("+arguments("Pt")+")" )
gROOT.ProcessLine( "macroPlot("+arguments("Eta")+")" )
gROOT.ProcessLine( "macroPlot("+arguments("PhiPlus")+")" )

# Resolution
# ----------
gROOT.ProcessLine(".x "+macrosDir+"ResolDraw.cc+(\"0\")")
gROOT.ProcessLine(".x "+macrosDir+"ResolDraw.cc+(\"3\")")
gROOT.ProcessLine(".x "+macrosDir+"ResolCompare.cc")

#os.system("root -l "+macrosDir+"ResolCompare.cc")

# Mass vs mass probability
gROOT.ProcessLine(".x "+macrosDir+"Plot_mass.C+(\"0_MuScleFit.root\", \"3_MuScleFit.root\")")
os.system("root -l "+macrosDir+"ShowMassComparison.C")

