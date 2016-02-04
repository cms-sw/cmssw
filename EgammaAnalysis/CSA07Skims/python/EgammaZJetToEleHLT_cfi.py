import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaZJetToEleHLT = copy.deepcopy(hltHighLevel)
EgammaZJetToEleHLT.HLTPaths = ['HLT_DoubleIsoEle10_L1I', 'HLT_DoubleIsoEle12_L1R']

