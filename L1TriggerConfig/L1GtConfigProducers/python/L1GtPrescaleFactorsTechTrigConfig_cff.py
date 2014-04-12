import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtPrescaleFactorsTechTrig_cfi import *
#
L1GtPrescaleFactorsTechTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


