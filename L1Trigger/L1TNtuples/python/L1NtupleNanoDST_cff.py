import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1EventTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *
from L1Trigger.L1TNtuples.l1uGTTree_cfi import *
from L1Trigger.L1TNtuples.l1uGTTestcrateTree_cfi import *

# use L1 objects from unpacked uGT output
l1UpgradeTree.egToken = cms.untracked.InputTag("gtStage2Digis","EGamma")
l1UpgradeTree.tauTokens = cms.untracked.VInputTag(cms.InputTag("gtStage2Digis","Tau"))
l1UpgradeTree.jetToken = cms.untracked.InputTag("gtStage2Digis","Jet")
l1UpgradeTree.muonToken = cms.untracked.InputTag("gtStage2Digis","Muon")
l1UpgradeTree.sumToken = cms.untracked.InputTag("gtStage2Digis","EtSum")

L1NtupleNANO = cms.Sequence(
  l1EventTree
  +l1UpgradeTree
  +l1uGTTestcrateTree
  +l1uGTTree
)
