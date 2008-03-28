import FWCore.ParameterSet.Config as cms

L1TriggerConfigOnline = cms.ESProducer("L1TriggerConfigOnlineProd",
    onlineAuthentication = cms.string('??'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('??')
)


