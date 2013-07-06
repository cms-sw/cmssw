import FWCore.ParameterSet.Config as cms

ecalEndcapLaserTask = cms.EDAnalyzer("EELaserTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE"),
    laserWavelengths = cms.untracked.vint32(1, 2, 3, 4)
)

