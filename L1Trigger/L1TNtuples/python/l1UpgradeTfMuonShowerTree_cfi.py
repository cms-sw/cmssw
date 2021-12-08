import FWCore.ParameterSet.Config as cms

l1UpgradeTfMuonShowerTree = cms.EDAnalyzer(
    "L1UpgradeTfMuonShowerTreeProducer",
    emtfMuonShowerToken = cms.untracked.InputTag("simEmtfShowers","EMTF"),
    maxL1UpgradeTfMuonShower = cms.uint32(12),
)

from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify( l1UpgradeTfMuonShowerTree,
    emtfMuonShowerToken = "none",
)
