#!/usr/bin/env python

""" This is script runs the resolution macros
"""

import os

from ROOT import gROOT

firstFile = "\"0\""
secondFile = "\"2\""
resonanceType = "JPsi"

macrosDir = os.popen("echo $CMSSW_BASE", "r").read().strip()
macrosDir += "/src/MuonAnalysis/MomentumScaleCalibration/test/Macros/"

# Resolution
# ----------
# The second parameter is a bool defining whether it should do half eta
# The third parameter is an integer defining the minimum number of entries required to perform a fit
gROOT.ProcessLine(".x "+macrosDir+"ResolDraw.cc++("+firstFile+", false, 100, 1, 2)")
gROOT.ProcessLine(".x "+macrosDir+"ResolDraw.cc++("+secondFile+", false, 100, 1, 2)")
gROOT.ProcessLine(".x "+macrosDir+"ResolCompare.cc("+firstFile+", "+secondFile+", true)")
