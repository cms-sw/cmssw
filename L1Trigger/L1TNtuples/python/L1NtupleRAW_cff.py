import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1Tree_cfi import *
from L1Trigger.L1TNtuples.l1ExtraTree_cfi import *
from L1Trigger.L1TNtuples.l1CaloTowerTree_cfi import *
from L1Trigger.L1TNtuples.l1MenuTree_cfi import *
from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import *

L1NtupleRAW = cms.Sequence(
  l1Tree
  +l1ExtraTree
  +l1CaloTowerTree
  +l1UpgradeTree
#  +l1MenuTree
)

