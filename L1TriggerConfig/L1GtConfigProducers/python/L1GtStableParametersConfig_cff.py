import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtStableParameters_cfi import *
#
L1GtStableParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtStableParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


