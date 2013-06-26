import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaLowEtTrigger = copy.deepcopy(hltHighLevel)
EgammaLowEtTrigger.HLTPaths = ['HLT_IsoEle18_L1R']

