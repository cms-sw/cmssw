import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtParameters_cfi import *
#
L1GtParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


