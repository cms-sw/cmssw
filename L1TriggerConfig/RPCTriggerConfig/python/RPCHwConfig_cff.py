import FWCore.ParameterSet.Config as cms

l1RPCHwConfig = cms.ESProducer("RPCTriggerHwConfig",
    disableTowers = cms.vint32(),
    disableCrates = cms.vint32(),
    disableTowersInCrates = cms.vint32(),
    enableTowers = cms.vint32(),
    enableCrates = cms.vint32(),
    enableTowersInCrates = cms.vint32(),
    disableAll = cms.bool(False)
)

rpchwconfsrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCHwConfigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

