import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.GMTConfigProducers.L1MuGMTParameters_cfi import *
L1MuGMTParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuGMTParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


