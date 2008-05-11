import FWCore.ParameterSet.Config as cms

ecalBarrelLaserTask = cms.EDFilter("EBLaserTask",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    prefixME = cms.untracked.string('EcalBarrel'),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker")
)


