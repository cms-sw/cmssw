import FWCore.ParameterSet.Config as cms

ecalEndcapTestPulseTask = cms.EDAnalyzer("EETestPulseTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EEDigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalGlobalUncalibRecHit","EcalUncalibRecHitsEE"),
    MGPAGains = cms.untracked.vint32(1, 6, 12),
    MGPAGainsPN = cms.untracked.vint32(1, 16)
)

