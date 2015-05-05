import FWCore.ParameterSet.Config as cms

### CCLA include latest RCT calibrations from UCT
from L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff import *
RCTConfigProducers.hOeCut = cms.double(999)

from L1Trigger.L1TCalorimeter.caloStage1Params_cfi import *
from Configuration.StandardSequences.RawToDigi_Repacked_cff import ecalDigis, hcalDigis

# RCT
from L1Trigger.Configuration.SimL1Emulator_cff import simRctDigis
simRctDigis.ecalDigis = cms.VInputTag(cms.InputTag('ecalDigis:EcalTriggerPrimitives'))
simRctDigis.hcalDigis = cms.VInputTag(cms.InputTag('hcalDigis'))

# stage 1 itself
from L1Trigger.L1TCalorimeter.L1TCaloStage1_cff import *
simRctUpgradeFormatDigis.regionTag = cms.InputTag("simRctDigis")
simRctUpgradeFormatDigis.emTag = cms.InputTag("simRctDigis")
simCaloStage1Digis.FirmwareVersion = cms.uint32(1) # 1=HI algos, 2=PP al
caloStage1Params.jetSeedThreshold = cms.double(0.)

# GT
from L1Trigger.Configuration.SimL1Emulator_cff import simGtDigis
simGtDigis.GmtInputTag = 'gtDigis'
simGtDigis.GctInputTag = 'caloStage1LegacyFormatDigis'
simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )

# the sequence
L1TCaloStage1_HIFromRaw = cms.Sequence(
    ecalDigis
    +hcalDigis
    +simRctDigis
    +L1TCaloStage1
    +simGtDigis
)
