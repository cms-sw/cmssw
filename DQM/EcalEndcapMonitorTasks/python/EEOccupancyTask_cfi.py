import FWCore.ParameterSet.Config as cms

ecalEndcapOccupancyTask = cms.EDFilter("EEOccupancyTask",
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    enableCleanup = cms.untracked.bool(True),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker")
)


