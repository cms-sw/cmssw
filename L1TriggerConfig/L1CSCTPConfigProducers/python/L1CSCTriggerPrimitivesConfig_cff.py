import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cfi import *
l1csctpconfsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1CSCTPParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(0)
)


