import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaWJetToMuHLT = copy.deepcopy(hltHighLevel)
EgammaWJetToMuHLT.HLTPaths = ['HLT_IsoMu11', 'HLT_Mu15_L1Mu7']

