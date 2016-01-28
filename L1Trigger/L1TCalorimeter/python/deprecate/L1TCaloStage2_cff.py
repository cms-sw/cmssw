import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloStage2Layer1Digis_cfi import *
from L1Trigger.L1TCalorimeter.caloStage2Digis_cfi import *

L1TCaloStage2 = cms.Sequence(
    caloStage2Layer1Digis +
    caloStage2Digis
)
