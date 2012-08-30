import FWCore.ParameterSet.Config as cms

ecalEndcapLaserTask = cms.EDAnalyzer("EELaserTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EEDigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    laserWavelengths = cms.untracked.vint32(1, 2, 3, 4)
)

