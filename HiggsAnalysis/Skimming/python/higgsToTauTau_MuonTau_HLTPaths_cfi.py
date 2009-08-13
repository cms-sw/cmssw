import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
higgsToTauTauMuonTauHLTFilter = copy.deepcopy(hltHighLevel)
higgsToTauTauElectronTauHLTFilter.TriggerResultsTag  = cms.InputTag("TriggerResults","","HLT8E29")

higgsToTauTauMuonTauHLTFilter.HLTPaths = ['HLT_Mu9']

