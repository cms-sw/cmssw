import FWCore.ParameterSet.Config as cms

#
#
#  Collection of all outputModules for QCD Analysis
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
#  Andreas Oehler's skims
from QCDAnalysis.Skimming.qcdJetFilterStreamHiOutputModule_cfi import *
from QCDAnalysis.Skimming.qcdJetFilterStreamMedOutputModule_cfi import *
from QCDAnalysis.Skimming.qcdJetFilterStreamLoOutputModule_cfi import *
# UE analysis QCD skims
from QCDAnalysis.Skimming.softJetsOutputModule_cfi import *
from QCDAnalysis.Skimming.diMuonOutputModule_cfi import *

