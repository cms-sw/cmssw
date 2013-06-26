import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
diffWToMuNuHLTFilter = copy.deepcopy(hltHighLevel)
diffWToMuNuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
diffWToMuNuHLTFilter.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso']

