import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
tauTo3MuHLTFilter = copy.deepcopy(hltHighLevel)
tauTo3MuHLTFilter.HLTPaths = ['HLT_IsoMu11', 'HLT_Mu15_L1Mu7', 'HLT_DoubleMu3', 'HLT_TripleMu3']

