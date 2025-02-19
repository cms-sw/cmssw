import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1ScalesProducers.L1MuGMTScales_cfi import *
L1MuGMTScalesRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuGMTScalesRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


