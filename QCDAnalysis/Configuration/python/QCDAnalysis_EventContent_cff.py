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
#  Andreas Oehler's 3 skims + 2 more from UE 8/21/07
#
#
#  Andreas Oehler's skims
from QCDAnalysis.Skimming.qcdJetFilterStreamHi_EventContent_cff import *
from QCDAnalysis.Skimming.qcdJetFilterStreamMed_EventContent_cff import *
from QCDAnalysis.Skimming.qcdJetFilterStreamLo_EventContent_cff import *
# QCD UE analysis skims
from QCDAnalysis.Skimming.diMuonEventContent_cfi import *
from QCDAnalysis.Skimming.softJetsEventContent_cfi import *

