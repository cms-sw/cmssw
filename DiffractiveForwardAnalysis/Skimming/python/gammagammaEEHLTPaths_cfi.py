import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
gammagammaEEHLTFilter = copy.deepcopy(hltHighLevel)
gammagammaEEHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
gammagammaEEHLTFilter.HLTPaths = ['CandHLT2ElectronExclusive']

