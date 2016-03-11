import FWCore.ParameterSet.Config as cms

l1UpgradeTree = cms.EDAnalyzer(
    "L1UpgradeTreeProducer",
    egToken = cms.untracked.InputTag("caloStage2Digis","EGamma"),
    tauTokens = cms.untracked.VInputTag(cms.InputTag("caloStage2Digis","Tau")),
    jetToken = cms.untracked.InputTag("caloStage2Digis","Jet"),
    muonToken = cms.untracked.InputTag("caloStage2Digis","Muon"),
    sumToken = cms.untracked.InputTag("caloStage2Digis","EtSum"),
    maxL1Upgrade = cms.uint32(60)
)

from Configuration.StandardSequences.Eras import eras

if eras.stage1L1Trigger.isChosen() or eras.Run2_25ns.isChosen():
    l1UpgradeTree.egToken = "caloStage1FinalDigis"
    l1UpgradeTree.tauTokens = cms.untracked.VInputTag("caloStage1FinalDigis:rlxTaus")
    l1UpgradeTree.jetToken = "caloStage1FinalDigis"
    l1UpgradeTree.muonToken = "none"
    l1UpgradeTree.sumToken = "caloStage1FinalDigis"
