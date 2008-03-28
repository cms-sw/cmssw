import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
zToTauTau_MuTauHLTFilter = copy.deepcopy(hltHighLevel)
zToTauTau_MuTauHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLTXMuonTau']

