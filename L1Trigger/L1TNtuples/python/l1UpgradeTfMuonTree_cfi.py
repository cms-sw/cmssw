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

from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( l1UpgradeTfMuonTree,
    bmtfMuonToken = "none",
    bmtfInputPhMuonToken = "none",
    bmtfInputThMuonToken = "none",
    omtfMuonToken = "none",
    emtfMuonToken = "none",
)
