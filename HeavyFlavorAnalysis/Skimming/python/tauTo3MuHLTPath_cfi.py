import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
tauTo3MuHLTFilter = copy.deepcopy(hltHighLevel)
tauTo3MuHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso', 'HLT2MuonNonIso', 'HLTNMuonNonIso']

