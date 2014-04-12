import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1ScalesProducers.l1CaloScales_cfi import *
emrcdsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1EmEtScaleRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

jetrcdsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1JetEtScaleRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

htmrcdsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1HtMissScaleRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

hfrrcdsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1HfRingEtScaleRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


