import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
zToMuMuHLTFilter = copy.deepcopy(hltHighLevel)
zToMuMuHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT2MuonNonIso']

