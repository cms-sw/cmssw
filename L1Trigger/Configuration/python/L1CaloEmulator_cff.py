import FWCore.ParameterSet.Config as cms

# RCT
from L1Trigger.RegionalCaloTrigger.rctDigis_cfi import *
# GCT
from L1Trigger.GlobalCaloTrigger.gctDigis_cfi import *
L1CaloEmulator = cms.Sequence(rctDigis*gctDigis)

# foo bar baz
# dSl7HD2CFnz6k
# oYeVHFk6xT8XP
