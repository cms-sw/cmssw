import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
zToMuMuRECOHLTFilter = copy.deepcopy(hltHighLevel)
zToMuMuRECOHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT2MuonNonIso', 'HLT2MuonZ']

