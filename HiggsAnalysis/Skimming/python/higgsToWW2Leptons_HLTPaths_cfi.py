import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToWW2LeptonsHLTFilter = copy.deepcopy(hltHighLevel)
higgsToWW2LeptonsHLTFilter.HLTPaths = ['HLT_IsoMu11', 'HLT_Mu15_L1Mu7', 'HLT_IsoEle15_L1I', 'HLT_IsoEle18_L1R', 'HLT_DoubleMu3', 
    'HLT_DoubleIsoEle10_L1I', 'HLT_DoubleIsoEle12_L1R', 'HLT_IsoEle8_IsoMu7', 'HLT_IsoEle10_Mu10_L1R']

