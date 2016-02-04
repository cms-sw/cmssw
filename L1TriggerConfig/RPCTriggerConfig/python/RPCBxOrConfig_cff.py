import FWCore.ParameterSet.Config as cms

l1RPCBxOrConfig = cms.ESProducer("RPCTriggerBxOrConfig",
    lastBX = cms.int32(0),
    firstBX = cms.int32(0)
)

rpcBxOrConfSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCBxOrConfigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

