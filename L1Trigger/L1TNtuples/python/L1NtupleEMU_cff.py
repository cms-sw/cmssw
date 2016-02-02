import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1CaloTowerTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *

l1CaloTowerEmuTree = l1CaloTowerTree.clone()
l1CaloTowerEmuTree.ecalToken = cms.untracked.InputTag("none")
l1CaloTowerEmuTree.hcalToken = cms.untracked.InputTag("none")
l1CaloTowerEmuTree.l1TowerToken = cms.untracked.InputTag("simCaloStage2Layer1Digis")
l1CaloTowerEmuTree.l1ClusterToken = cms.untracked.InputTag("simCaloStage2Digis", "MP")

l1UpgradeEmuTree = l1UpgradeTree.clone()
l1UpgradeEmuTree.egToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.tauToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.jetToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.muonToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.sumToken = cms.untracked.InputTag("simCaloStage2Digis")

L1NtupleEMU = cms.Sequence(
  l1CaloTowerEmuTree
  +l1UpgradeEmuTree
)
