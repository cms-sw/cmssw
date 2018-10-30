import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1EventTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *
from L1Trigger.L1TNtuples.l1uGTTree_cfi import *

l1uGTTree.ugtToken = cms.InputTag("hltGtStage2Digis")

l1UpgradeTree.egToken = cms.untracked.InputTag("hltGtStage2Digis","EGamma")
l1UpgradeTree.tauTokens = cms.untracked.VInputTag(cms.InputTag("hltGtStage2Digis","Tau"))
l1UpgradeTree.jetToken = cms.untracked.InputTag("hltGtStage2Digis","Jet")
l1UpgradeTree.muonToken = cms.untracked.InputTag("hltGtStage2Digis","Muon")
l1UpgradeTree.sumToken = cms.untracked.InputTag("hltGtStage2Digis","EtSum")

L1NtupleNANO = cms.Sequence(
  l1EventTree
  +l1UpgradeTree
  +l1uGTTree
)
