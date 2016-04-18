import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Stage2 Emulator Modules

# EMTF
from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *

valEmtfStage2Digis = simEmtfDigis.clone()
valEmtfStage2Digis.CSCInput = "csctfDigis"

#-------------------------------------------------
# Emulator DQM Modules

# EMTF
from DQM.L1TMonitor.L1TdeStage2EMTF_cfi import *

#-------------------------------------------------
# Stage2 Emulator and Emulator DQM Sequences

l1tStage2Emulator = cms.Sequence(
    valEmtfStage2Digis
)

l1tStage2EmulatorOnlineDQM = cms.Sequence(
    l1tdeStage2Emtf
)

