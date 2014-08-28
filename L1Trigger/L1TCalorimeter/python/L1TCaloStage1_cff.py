import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.rctUpgradeFormatDigis_cfi import *
from L1Trigger.L1TCalorimeter.caloStage1Digis_cfi import *
from L1Trigger.L1TCalorimeter.caloStage1FinalDigis_cfi import *
from L1Trigger.L1TCalorimeter.caloStage1LegacyFormatDigis_cfi import *

L1TCaloStage1 = cms.Sequence(
    rctUpgradeFormatDigis +
    caloStage1Digis +
    caloStage1FinalDigis +
    caloStage1LegacyFormatDigis
)
