import FWCore.ParameterSet.Config as cms

#
#
#  Collection of all Skim Paths for QCD Analysis
#
#
#
#
#  Added by D. Mason 8/7/07
#
#  Pulled out single jet skims -- brought in 
#  Andreas Oehler's 3 skims + 2 more from UE  8/21/07
#
#
#  Andreas Oehler's skims -- apparently requires random numbers...
from QCDAnalysis.Skimming.qcdJetFilterStreamHiPath_cff import *
from QCDAnalysis.Skimming.qcdJetFilterStreamMedPath_cff import *
from QCDAnalysis.Skimming.qcdJetFilterStreamLoPath_cff import *
# QCD UE analysis Skims
from QCDAnalysis.Skimming.softJetsPath_cff import *
from QCDAnalysis.Skimming.diMuonPath_cff import *

