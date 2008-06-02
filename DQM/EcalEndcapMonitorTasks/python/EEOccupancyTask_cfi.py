import FWCore.ParameterSet.Config as cms

ecalEndcapOccupancyTask = cms.EDFilter("EEOccupancyTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker")
)


