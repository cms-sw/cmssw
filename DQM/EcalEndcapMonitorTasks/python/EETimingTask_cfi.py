import FWCore.ParameterSet.Config as cms

ecalEndcapTimingTask = cms.EDFilter("EETimingTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEE")
)

