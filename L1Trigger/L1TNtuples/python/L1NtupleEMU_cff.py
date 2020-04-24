import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1CaloTowerTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTfMuonTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *
from L1Trigger.L1TNtuples.l1EventTree_cfi import *
from L1Trigger.L1TNtuples.l1uGTTree_cfi import *

l1UpgradeTfMuonEmuTree = l1UpgradeTfMuonTree.clone()
l1UpgradeTfMuonEmuTree.bmtfMuonToken = cms.untracked.InputTag("simBmtfDigis","BMTF") 
l1UpgradeTfMuonEmuTree.omtfMuonToken = cms.untracked.InputTag("simOmtfDigis","OMTF") 
l1UpgradeTfMuonEmuTree.emtfMuonToken = cms.untracked.InputTag("simEmtfDigis","EMTF") 

l1CaloTowerEmuTree = l1CaloTowerTree.clone()
l1CaloTowerEmuTree.ecalToken = cms.untracked.InputTag("simEcalTriggerPrimitiveDigis")
l1CaloTowerEmuTree.hcalToken = cms.untracked.InputTag("simHcalTriggerPrimitiveDigis")
l1CaloTowerEmuTree.l1TowerToken = cms.untracked.InputTag("simCaloStage2Layer1Digis")
l1CaloTowerEmuTree.l1ClusterToken = cms.untracked.InputTag("simCaloStage2Digis", "MP")

l1UpgradeEmuTree = l1UpgradeTree.clone()
l1UpgradeEmuTree.egToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.tauTokens = cms.untracked.VInputTag("simCaloStage2Digis")
l1UpgradeEmuTree.jetToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.muonToken = cms.untracked.InputTag("simGmtStage2Digis")
#l1UpgradeEmuTree.muonToken = cms.untracked.InputTag("muonLegacyInStage2FormatDigis")
l1UpgradeEmuTree.sumToken = cms.untracked.InputTag("simCaloStage2Digis")

#l1legacyMuonEmuTree = l1UpgradeTree.clone()
#l1legacyMuonEmuTree.muonToken = cms.untracked.InputTag("muonLegacyInStage2FormatDigis","imdMuonsLegacy")

l1uGTEmuTree = l1uGTTree.clone()
l1uGTEmuTree.ugtToken = cms.InputTag("simGtStage2Digis")

from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify(l1UpgradeEMUTree,
    egToken = "simCaloStage1FinalDigis",
    tauTokens = cms.untracked.VInputTag("simCaloStage1FinalDigis:rlxTaus"),
    jetToken = "simCaloStage1FinalDigis",
    muonToken = "simGtDigis",
    sumToken = "simCaloStage1FinalDigis",
)

L1NtupleEMU = cms.Sequence(
  l1EventTree
  +l1UpgradeTfMuonEmuTree
  +l1CaloTowerEmuTree
  +l1UpgradeEmuTree
#  +l1MuonEmuTree
  +l1uGTEmuTree
)
