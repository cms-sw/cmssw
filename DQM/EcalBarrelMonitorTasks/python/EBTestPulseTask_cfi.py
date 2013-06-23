import FWCore.ParameterSet.Config as cms

ecalBarrelTestPulseTask = cms.EDAnalyzer("EBTestPulseTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    MGPAGains = cms.untracked.vint32(1, 6, 12),
    MGPAGainsPN = cms.untracked.vint32(1, 16)
)

