import FWCore.ParameterSet.Config as cms

ecalBarrelOccupancyTask = cms.EDFilter("EBOccupancyTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker")
)


