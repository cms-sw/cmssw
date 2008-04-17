import FWCore.ParameterSet.Config as cms

L1TriggerKeyOnline = cms.ESProducer("L1TriggerKeyOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    tscKey = cms.string('dummy'),
    onlineDB = cms.string('oracle://omds/GLOBALCALTRIGGER')
)


