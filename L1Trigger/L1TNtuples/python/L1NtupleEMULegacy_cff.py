import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *
from L1Trigger.L1TNtuples.l1EventTree_cfi import *

l1legacyMuonEmuTree = l1UpgradeTree.clone()
l1legacyMuonEmuTree.muonToken = cms.untracked.InputTag("muonLegacyInStage2FormatDigis","imdMuonsLegacy")

L1NtupleEMULegacy = cms.Sequence(
  l1EventTree
  +l1legacyMuonEmuTree
)

