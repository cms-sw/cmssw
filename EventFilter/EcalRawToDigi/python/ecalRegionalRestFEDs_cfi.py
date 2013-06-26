import FWCore.ParameterSet.Config as cms

ecalRegionalRestFEDs = cms.EDProducer("EcalRawToRecHitRoI",
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    type = cms.string('all')
)


