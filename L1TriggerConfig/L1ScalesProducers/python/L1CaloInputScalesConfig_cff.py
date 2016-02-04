import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1ScalesProducers.L1CaloInputScales_cfi import *
l1CaloEcalScaleRecord = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1CaloEcalScaleRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1CaloHcalScaleRecord = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1CaloHcalScaleRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


