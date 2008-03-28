import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtFactors_cfi import *
#
L1GtPrescaleFactorsRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtPrescaleFactorsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

L1GtTriggerMaskRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


