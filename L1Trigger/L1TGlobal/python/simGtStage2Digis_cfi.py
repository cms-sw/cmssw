#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

# cfi uGT emulator

simGtStage2Digis = cms.EDProducer("L1TGlobalProducer",
    GmtInputTag = cms.InputTag("simGmtStage2Digis"),
    ExtInputTag = cms.InputTag("none"),
    CaloInputTag = cms.InputTag("simCaloStage2Digis"),
    AlgorithmTriggersUnmasked = cms.bool(False),    
    AlgorithmTriggersUnprescaled = cms.bool(False),
)

