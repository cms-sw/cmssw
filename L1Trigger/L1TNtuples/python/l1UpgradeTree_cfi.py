import FWCore.ParameterSet.Config as cms

l1UpgradeTree = cms.EDAnalyzer(
    "L1UpgradeTreeProducer",
    egToken = cms.untracked.InputTag("caloStage2Digis"),
    tauTokens = cms.untracked.VInputTag("caloStage2Digis"),
    jetToken = cms.untracked.InputTag("caloStage2Digis"),
    muonToken = cms.untracked.InputTag("gmtStage2Digis"),
    sumToken = cms.untracked.InputTag("caloStage2Digis"),
    maxL1Upgrade = cms.uint32(60)
)

from Configuration.StandardSequences.Eras import eras

if eras.stage1L1Trigger.isChosen() or eras.Run2_25ns.isChosen():
    l1UpgradeTree.egToken = "caloStage1FinalDigis"
    l1UpgradeTree.tauTokens = cms.untracked.VInputTag("caloStage1FinalDigis:rlxTaus")
    l1UpgradeTree.jetToken = "caloStage1FinalDigis"
    l1UpgradeTree.muonToken = "none"
    l1UpgradeTree.sumToken = "caloStage1FinalDigis"
