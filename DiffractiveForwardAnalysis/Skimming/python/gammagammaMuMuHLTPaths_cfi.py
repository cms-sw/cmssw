import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
gammagammaMuMuHLTFilter = copy.deepcopy(hltHighLevel)
gammagammaMuMuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
gammagammaMuMuHLTFilter.HLTPaths = ['HLT2MuonNonIso']

