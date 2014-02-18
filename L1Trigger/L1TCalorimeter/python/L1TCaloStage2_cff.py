import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.l1tCaloStage2TowerDigis_cfi import *
from L1Trigger.L1TCalorimeter.l1tCaloStage2Digis_cfi import *

L1TCaloStage2 = cms.Sequence(
    l1tCaloStage2TowerDigis *
    l1tCaloStage2Digis
)
