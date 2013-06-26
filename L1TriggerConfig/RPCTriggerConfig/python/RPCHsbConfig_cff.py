import FWCore.ParameterSet.Config as cms

l1RPCHsbConfig = cms.ESProducer("RPCTriggerHsbConfig",
    hsb0Mask = cms.vint32(3, 3, 3, 3, 3, 3, 3, 3),
    hsb1Mask = cms.vint32(3, 3, 3, 3, 3, 3, 3, 3)
)

rpchsbConfSrc = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1RPCHsbConfigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

