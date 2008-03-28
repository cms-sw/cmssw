import FWCore.ParameterSet.Config as cms

ecalBarrelOccupancyTask = cms.EDFilter("EBOccupancyTask",
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    enableCleanup = cms.untracked.bool(True),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker")
)


