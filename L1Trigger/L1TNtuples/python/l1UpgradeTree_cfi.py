import FWCore.ParameterSet.Config as cms

l1UpgradeTree = cms.EDAnalyzer(
    "L1UpgradeTreeProducer",
    egToken = cms.untracked.InputTag("caloStage2Digis","EGamma"),
    tauTokens = cms.untracked.VInputTag(cms.InputTag("caloStage2Digis","Tau")),
    jetToken = cms.untracked.InputTag("caloStage2Digis","Jet"),
    muonToken = cms.untracked.InputTag("gmtStage2Digis","Muon"),
    muonShowerToken = cms.untracked.InputTag("simGmtShowerDigis"),
    muonLegacyToken = cms.untracked.InputTag("muonLegacyInStage2FormatDigis","legacyMuon"),
    sumToken = cms.untracked.InputTag("caloStage2Digis","EtSum"),
    sumZDCToken = cms.untracked.InputTag("zdcEtSumProducer", "zdcEtSums"),
    maxL1Upgrade = cms.uint32(60)
)

from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( l1UpgradeTree,
    egToken = "caloStage1FinalDigis",
    tauTokens = cms.untracked.VInputTag("caloStage1FinalDigis:rlxTaus"),
    jetToken = "caloStage1FinalDigis",
    muonToken = "muonLegacyInStage2FormatDigis",
    muonShowerToken = "",
    sumToken = "caloStage1FinalDigis",
)
