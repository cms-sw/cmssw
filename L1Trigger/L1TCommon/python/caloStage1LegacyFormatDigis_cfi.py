import FWCore.ParameterSet.Config as cms

caloStage1LegacyFormatDigis = cms.EDProducer(
    "l1t::L1TCaloUpgradeToGCTConverter",
    InputCollection = cms.InputTag("TESTcaloStage1Digis"),
    InputRlxTauCollection = cms.InputTag("TESTcaloStage1Digis:rlxTaus"),
    InputIsoTauCollection = cms.InputTag("TESTcaloStage1Digis:isoTaus"),
    InputHFSumsCollection = cms.InputTag("TESTcaloStage1Digis:HFRingSums"),
    InputHFCountsCollection = cms.InputTag("TESTcaloStage1Digis:HFBitCounts")
)
