import FWCore.ParameterSet.Config as cms

l1UpgradeTree = cms.EDAnalyzer(
    "L1UpgradeTreeProducer",
    egToken = cms.untracked.InputTag("caloStage2Digis"),
    tauTokens = cms.untracked.VInputTag("caloStage2Digis"),
    jetToken = cms.untracked.InputTag("caloStage2Digis"),
    muonToken = cms.untracked.InputTag("caloStage2Digis"),
    sumToken = cms.untracked.InputTag("caloStage2Digis"),
    maxL1Upgrade = cms.uint32(60)
)

