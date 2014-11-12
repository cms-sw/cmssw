import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.simRctUpgradeFormatDigis_cfi import *
from L1Trigger.L1TCalorimeter.simCaloStage1Digis_cfi import *
from L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi import *
from L1Trigger.L1TCalorimeter.simCaloStage1LegacyFormatDigis_cfi import *

L1TCaloStage1 = cms.Sequence(
    simRctUpgradeFormatDigis +
    simCaloStage1Digis +
    simCaloStage1FinalDigis +
    simCaloStage1LegacyFormatDigis
)
