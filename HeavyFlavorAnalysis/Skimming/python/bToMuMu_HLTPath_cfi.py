import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
bToMuMuHLTFilter = copy.deepcopy(hltHighLevel)
bToMuMuHLTFilter.HLTPaths = ['HLT2MuonNonIso', 'HLTBJPsiMuMu', 'HLT2MuonSameSign']

