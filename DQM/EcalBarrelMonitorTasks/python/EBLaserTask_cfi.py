import FWCore.ParameterSet.Config as cms

ecalBarrelLaserTask = cms.EDAnalyzer("EBLaserTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EBDigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEB"),
    laserWavelengths = cms.untracked.vint32(1, 2, 3, 4),
    filterEmptyEvents = cms.untracked.bool(True)
)

