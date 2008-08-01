import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.GctConfigProducers.l1GctConfig_cfi import *
l1GctParamsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetFinderParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1GctConfigRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetCalibFunRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1GctJcPosParsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetCounterPositiveEtaRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

l1GctJcNegParsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetCounterNegativeEtaRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


