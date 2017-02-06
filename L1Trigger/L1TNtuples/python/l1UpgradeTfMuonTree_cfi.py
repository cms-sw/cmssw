import FWCore.ParameterSet.Config as cms

l1UpgradeTfMuonTree = cms.EDAnalyzer(
    "L1UpgradeTfMuonTreeProducer",
    bmtfMuonToken = cms.untracked.InputTag("bmtfDigis","BMTF"),
    bmtfInputPhMuonToken = cms.untracked.InputTag("bmtfDigis",""),
    bmtfInputThMuonToken = cms.untracked.InputTag("bmtfDigis",""),
    omtfMuonToken = cms.untracked.InputTag("omtfDigis","OMTF"),
    emtfMuonToken = cms.untracked.InputTag("emtfDigis","EMTF"),
    maxL1UpgradeTfMuon = cms.uint32(60)
)

from Configuration.StandardSequences.Eras import eras

if eras.stage1L1Trigger.isChosen() or eras.Run2_25ns.isChosen():
    l1UpgradeTree.bmtfMuonToken = "none"
    l1UpgradeTree.bmtfInputPhMuonToken = "none"
    l1UpgradeTree.bmtfInputThMuonToken = "none"
    l1UpgradeTree.omtfMuonToken = "none"
    l1UpgradeTree.emtfMuonToken = "none"
