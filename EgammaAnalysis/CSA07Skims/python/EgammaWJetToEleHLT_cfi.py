import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaWJetToEleHLT = copy.deepcopy(hltHighLevel)
EgammaWJetToEleHLT.HLTPaths = ['HLT_IsoEle15_L1I', 'HLT_IsoEle18_L1R']

