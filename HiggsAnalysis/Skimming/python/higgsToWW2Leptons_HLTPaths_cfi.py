import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToWW2LeptonsHLTFilter = copy.deepcopy(hltHighLevel)
#higgsToWW2LeptonsHLTFilter.HLTPaths = ['HLT_IsoMu11', 'HLT_Mu15_L1Mu7', 'HLT_IsoEle15_L1I', 'HLT_IsoEle18_L1R', 'HLT_DoubleMu3', 
#    'HLT_DoubleIsoEle10_L1I', 'HLT_DoubleIsoEle12_L1R', 'HLT_IsoEle8_IsoMu7', 'HLT_IsoEle10_Mu10_L1R']

#Lum 8E29
#higgsToWW2LeptonsHLTFilter.HLTPaths = ['HLT_L1MuOpen','HLT_L1Mu','HLT_Mu3','HLT_IsoMu3',
#	'HLT_DoubleMu0','HLT_DoubleMu3','HLT_L1DoubleMuOpen',
#	'HLT_Ele10_LW_L1R','HLT_L1Mu14_L1SingleEG10']

#Lum 1E31
higgsToWW2LeptonsHLTFilter.HLTPaths = ['HLT_L1MuOpen','HLT_L1Mu','HLT_Mu5',
	'HLT_L1Mu20','HLT_L2Mu11','HLT_DoubleMu0',
	'HLT_L1SingleEG5','HLT_Ele10_SW_L1R','HLT_Ele15_SW_L1R',
	'HLT_Ele15_SW_LooseTrackIso_L1R']
