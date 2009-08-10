import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToTauTauMuonTauHLTFilter = copy.deepcopy(hltHighLevel)
higgsToTauTauMuonTauHLTFilter.HLTPaths = ['HLT_Mu15', 'HLT_IsoMu11']

