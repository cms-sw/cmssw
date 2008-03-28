import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
EgammaWJetToMuHLT = copy.deepcopy(hltHighLevel)
EgammaWJetToMuHLT.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso']

