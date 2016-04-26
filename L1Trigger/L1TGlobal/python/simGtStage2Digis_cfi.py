#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

# cfi uGT emulator

simGtStage2Digis = cms.EDProducer("L1TGlobalProducer",
    MuonInputTag = cms.InputTag("simGmtStage2Digis"),
    ExtInputTag = cms.InputTag("simGtExtFakeStage2Digis"),
    EGammaInputTag = cms.InputTag("simCaloStage2Digis"),
    TauInputTag = cms.InputTag("simCaloStage2Digis"),
    JetInputTag = cms.InputTag("simCaloStage2Digis"),
    EtSumInputTag = cms.InputTag("simCaloStage2Digis"),
    AlgorithmTriggersUnmasked = cms.bool(False),    
    AlgorithmTriggersUnprescaled = cms.bool(False),
)

