import FWCore.ParameterSet.Config as cms

ecalEndcapLedTask = cms.EDAnalyzer("EELedTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EEDigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    ledWavelengths = cms.untracked.vint32(1, 2)
)

