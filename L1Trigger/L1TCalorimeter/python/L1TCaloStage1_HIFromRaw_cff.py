import FWCore.ParameterSet.Config as cms

### CCLA include latest RCT calibrations from UCT
from L1Trigger.L1TCalorimeter.caloStage1RCTLuts_cff import *
RCTConfigProducers.eicIsolationThreshold = cms.uint32(7)
RCTConfigProducers.hOeCut = cms.double(999)
RCTConfigProducers.eMinForHoECut = cms.double(999)
RCTConfigProducers.eMaxForHoECut = cms.double(999)
RCTConfigProducers.hMinForHoECut = cms.double(999)
RCTConfigProducers.eMinForFGCut = cms.double(999)

from L1Trigger.L1TCalorimeter.caloStage1Params_cfi import *
caloStage1Params.jetSeedThreshold = cms.double(0.)
caloStage1Params.regionPUSType = cms.string("zeroWall")

from Configuration.StandardSequences.RawToDigi_Repacked_cff import ecalDigis, hcalDigis

# RCT
from L1Trigger.Configuration.SimL1Emulator_cff import simRctDigis
simRctDigis.ecalDigis = cms.VInputTag(cms.InputTag('ecalDigis:EcalTriggerPrimitives'))
simRctDigis.hcalDigis = cms.VInputTag(cms.InputTag('hcalDigis'))

# stage 1 itself
from L1Trigger.L1TCalorimeter.L1TCaloStage1_cff import *
simRctUpgradeFormatDigis.regionTag = cms.InputTag("simRctDigis")
simRctUpgradeFormatDigis.emTag = cms.InputTag("simRctDigis")

# GT
from L1Trigger.Configuration.SimL1Emulator_cff import simGtDigis
simGtDigis.GmtInputTag = 'simGmtDigis'
simGtDigis.GctInputTag = 'simCaloStage1LegacyFormatDigis'
simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )

# the sequence
L1TCaloStage1_HIFromRaw = cms.Sequence(
    ecalDigis
    +hcalDigis
    +simRctDigis
    +L1TCaloStage1
    +simGtDigis
)
