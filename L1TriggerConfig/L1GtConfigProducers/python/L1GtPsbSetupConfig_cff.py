import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.l1GtPsbSetup_cfi import *
#
L1GtPsbSetupRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtPsbSetupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


