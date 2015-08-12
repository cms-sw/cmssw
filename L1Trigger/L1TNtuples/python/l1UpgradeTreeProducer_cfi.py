import FWCore.ParameterSet.Config as cms

l1UpgradeTreeProducer = cms.EDAnalyzer(
    "L1UpgradeTreeProducer",
    egLabel = cms.untracked.InputTag("caloStage2Digis"),
    tauLabel = cms.untracked.InputTag("caloStage2Digis"),
    jetLabel = cms.untracked.InputTag("caloStage2Digis"),
    muonLabel = cms.untracked.InputTag("caloStage2Digis"),
    sumsLabel = cms.untracked.InputTag("caloStage2Digis"),
    maxL1Upgrade = cms.uint32(20)
)

