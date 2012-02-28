import FWCore.ParameterSet.Config as cms

ecalEndcapTimingTask = cms.EDAnalyzer("EETimingTask",
    prefixME = cms.untracked.string('Ecal'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    useBeamStatus = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    L1GtEvmReadoutRecord = cms.InputTag("l1GtEvmUnpack"),
    energyThreshold = cms.untracked.double(3.0)
)

