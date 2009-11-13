import FWCore.ParameterSet.Config as cms
import copy

# Trigger
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
hltFilter = hltHighLevel.clone()
hltFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
hltFilter.HLTPaths = ['HLT_SingleLooseIsoTau20']
