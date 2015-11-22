import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1Tree_cfi import *
from L1Trigger.L1TNtuples.l1ExtraTree_cfi import *
from L1Trigger.L1TNtuples.l1CaloTowerTree_cfi import *
from L1Trigger.L1TNtuples.l1MenuTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *

l1CaloTowerSimTree = l1CaloTowerTree.clone()
l1CaloTowerSimTree.ecalToken = cms.untracked.InputTag("none")
l1CaloTowerSimTree.hcalToken = cms.untracked.InputTag("none")
l1CaloTowerSimTree.l1TowerToken = cms.untracked.InputTag("simCaloStage2Layer1Digis")

l1UpgradeSimTree = l1UpgradeTree.clone()
l1UpgradeSimTree.egToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeSimTree.tauToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeSimTree.jetToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeSimTree.muonToken = cms.untracked.InputTag("simCaloStage2Digis")
l1UpgradeSimTree.sumsToken = cms.untracked.InputTag("simCaloStage2Digis")

L1NtupleRAW = cms.Sequence(
  l1Tree
#  +l1ExtraTree
  +l1CaloTowerTree
  +l1CaloTowerSimTree
  +l1UpgradeTree
  +l1UpgradeSimTree
#  +l1MenuTree
)
