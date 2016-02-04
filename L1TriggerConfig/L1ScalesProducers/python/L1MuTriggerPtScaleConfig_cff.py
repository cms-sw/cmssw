import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScale_cfi import *
L1MuTriggerPtScaleRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuTriggerPtScaleRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)



