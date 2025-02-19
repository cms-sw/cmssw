import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.GctConfigProducers.l1GctConfig_cfi import *
l1GctParamsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetFinderParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1GctChanMaskRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctChannelMaskRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


