import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1EventTree_cfi import *
from L1Trigger.L1TNtuples.l1CaloTowerTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *
from L1Trigger.L1TNtuples.l1uGTTree_cfi import *

L1NtupleRAWCalo = cms.Sequence(
  l1EventTree
  +l1CaloTowerTree
  +l1UpgradeTree
  +l1uGTTree
)
