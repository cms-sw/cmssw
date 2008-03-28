import FWCore.ParameterSet.Config as cms

ecalRegionalRestFEDs = cms.EDFilter("EcalRawToRecHitRoI",
    sourceTag = cms.InputTag("EcalRawToRecHitFacility"),
    type = cms.string('all')
)


