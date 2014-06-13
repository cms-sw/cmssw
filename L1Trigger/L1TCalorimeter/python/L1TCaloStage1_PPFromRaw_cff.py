import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TCalorimeter.caloStage1Params_cfi import *

# HCAL TP hack
from L1Trigger.L1TCalorimeter.L1TRerunHCALTP_FromRaw_cff import *

# RCT
# HCAL input would be from hcalDigis if hack not needed
from L1Trigger.Configuration.SimL1Emulator_cff import simRctDigis
simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) )

# stage 1 itself
from L1Trigger.L1TCalorimeter.L1TCaloStage1_cff import *
rctUpgradeFormatDigis.regionTag = cms.InputTag("simRctDigis")
rctUpgradeFormatDigis.emTag = cms.InputTag("simRctDigis")

# GT
from L1Trigger.Configuration.SimL1Emulator_cff import simGtDigis
simGtDigis.GmtInputTag = 'gtDigis'
simGtDigis.GctInputTag = 'caloStage1LegacyFormatDigis'
simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )

# the sequence
L1TCaloStage1_PPFromRaw = cms.Sequence(
    L1TRerunHCALTP_FromRAW
    +simRctDigis
    +L1TCaloStage1
    +simGtDigis
)
