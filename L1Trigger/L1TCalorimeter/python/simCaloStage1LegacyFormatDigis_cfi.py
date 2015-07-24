import FWCore.ParameterSet.Config as cms

simCaloStage1LegacyFormatDigis = cms.EDProducer(
    "L1TCaloUpgradeToGCTConverter",
    InputCollection = cms.InputTag("simCaloStage1FinalDigis"),
    InputRlxTauCollection = cms.InputTag("simCaloStage1Digis:rlxTaus"),
    InputIsoTauCollection = cms.InputTag("simCaloStage1Digis:isoTaus"),
    InputHFSumsCollection = cms.InputTag("simCaloStage1Digis:HFRingSums"),
    InputHFCountsCollection = cms.InputTag("simCaloStage1Digis:HFBitCounts"),
    bxMin = cms.int32(-2),
    bxMax = cms.int32(2)
)
