import FWCore.ParameterSet.Config as cms

#Include configuration ParameterSets
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigParams_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigMap_cff import *

L1DTConfigFromDB = cms.ESProducer("DTConfigDBProducer",
    DTTPGMapBlock,
    DTTPGParametersBlock,
### 110202 SV CONFIGURATIONS, LUTs and ACCEPT. FROM DB
    cfgConfig  = cms.bool(False),
    bxOffset  = cms.int32(19),
    finePhase = cms.double(25.),
    TracoLutsFromDB = cms.bool(True),
    UseBtiAcceptParam = cms.bool(True),
    UseT0 = cms.bool(False),
### 110202 SV LUTs from geometry
#    TracoLutsFromDB = cms.bool(False),
### 110202 SV BTI trigger acceptance from geometry
#    UseBtiAcceptParam = cms.bool(False),
### 110202 SV CONFIGURATIONS FROM CFF files and luts and accept.from geometry
#    cfgConfig  = cms.bool(True),
#    TracoLutsFromDB = cms.bool(False),
#    UseBtiAcceptParam = cms.bool(False),
#    UseT0 = cms.bool(False),
#
    debugDB    = cms.bool(False),
    debugBti   = cms.int32(0),
    debugTraco = cms.int32(0),
    debugTSP   = cms.bool(False),
    debugTST   = cms.bool(False),
    debugTU    = cms.bool(False),
    debugSC    = cms.bool(False),
    debugLUTs  = cms.bool(False),
    debug      = cms.bool(False),
    debugPed   = cms.bool(False)
)
