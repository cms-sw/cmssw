import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScales_cfi import *
L1MuTriggerScalesRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuTriggerScalesRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


