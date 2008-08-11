import FWCore.ParameterSet.Config as cms

l1RPCHwConfig = cms.ESProducer("RPCTriggerHwConfig",
    disableCrates = cms.vint32(),
    enableTowers = cms.vint32(),
    enableCrates = cms.vint32(),
    disableAll = cms.bool(False),
    lastBX = cms.int32(0),
    firstBX = cms.int32(0),
    disableTowers = cms.vint32()
)

rpchwconfsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCHwConfigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

