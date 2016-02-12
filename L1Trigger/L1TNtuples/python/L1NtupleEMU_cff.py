import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1CaloTowerTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *
from L1Trigger.L1TNtuples.l1EventTree_cfi import *

l1CaloTowerEmuTree = l1CaloTowerTree.clone()
l1CaloTowerEmuTree.ecalToken = cms.untracked.InputTag("none")
l1CaloTowerEmuTree.hcalToken = cms.untracked.InputTag("simHcalTriggerPrimitiveDigis")
l1CaloTowerEmuTree.l1TowerToken = cms.untracked.InputTag("simCaloStage2Layer1Digis")
l1CaloTowerEmuTree.l1ClusterToken = cms.untracked.InputTag("simCaloStage2Digis", "MP")

l1UpgradeEmuTree = l1UpgradeTree.clone()
l1UpgradeEmuTree.egToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.tauTokens = cms.untracked.VInputTag("simCaloStage2Digis")
l1UpgradeEmuTree.jetToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeEmuTree.muonToken = cms.untracked.InputTag("simGmtStage2Digis")
l1UpgradeEmuTree.sumToken = cms.untracked.InputTag("simCaloStage2Digis")

if eras.stage1L1Trigger.isChosen() or eras.Run2_25ns.isChosen():
    l1UpgradeEMUTree.egToken = "simCaloStage1FinalDigis"
    l1UpgradeEMUTree.tauTokens = cms.untracked.VInputTag("simCaloStage1FinalDigis:rlxTaus")
    l1UpgradeEMUTree.jetToken = "simCaloStage1FinalDigis"
    l1UpgradeEMUTree.muonToken = "simGtDigis"
    l1UpgradeEMUTree.sumToken = "simCaloStage1FinalDigis"

L1NtupleEMU = cms.Sequence(
  l1CaloTowerEmuTree
  +l1UpgradeEmuTree
  +l1EventTree
)
