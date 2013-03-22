import FWCore.ParameterSet.Config as cms

ecalBarrelLaserTask = cms.EDAnalyzer("EBLaserTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    laserWavelengths = cms.untracked.vint32(1, 2, 3, 4)
)

