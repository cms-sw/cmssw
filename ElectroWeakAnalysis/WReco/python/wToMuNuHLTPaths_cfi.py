import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
wToMuNuHLTFilter = copy.deepcopy(hltHighLevel)
wToMuNuHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso']

