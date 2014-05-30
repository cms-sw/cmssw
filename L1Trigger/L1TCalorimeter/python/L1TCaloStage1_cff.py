import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.l1tStage1CaloParams_cfi import *
from L1Trigger.L1TCalorimeter.l1tCaloStage1Digis_cfi import *

from L1Trigger.Configuration.SimL1Emulator_cff import *

simGctDigis_Stage1 = cms.Sequence(
    rctStage1FormatDigis +
    caloStage1Digis +
    caloStage1FinalDigis +
    caloLegacyFormatDigis
)

simGtDigis.GctInputTag = 'caloLegacyFormatDigis'

SimL1Emulator_Stage1 = cms.Sequence(
    simRctDigis +
    simGctDigis_Stage1 +
    SimL1MuTriggerPrimitives +
    SimL1MuTrackFinders +
    simRpcTriggerDigis +
    simGmtDigis +
    SimL1TechnicalTriggers +
    simGtDigis )
