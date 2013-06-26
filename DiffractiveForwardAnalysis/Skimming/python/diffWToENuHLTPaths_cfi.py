import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
diffWToENuHLTFilter = copy.deepcopy(hltHighLevel)
diffWToENuHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
diffWToENuHLTFilter.HLTPaths = ['HLT1Electron']

