#!/usr/bin/env python

""" This is script runs all the macros in the local Macros dir
"""

import os

from ROOT import gROOT

firstFile = "\"Ideal\""
secondFile = "\"Fake\""
thirdFile = "\"Smear\""

resonanceType = "Z"

macrosDir = os.popen("echo $CMSSW_BASE", "r").read().strip()
macrosDir += "/src/MuonAnalysis/MomentumScaleCalibration/test/Macros/"

gROOT.ProcessLine(".x "+macrosDir+"ComparisonAfterSmearing.cc("+firstFile+", "+secondFile+", "+thirdFile+")")



