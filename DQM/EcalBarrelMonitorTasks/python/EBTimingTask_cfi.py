import FWCore.ParameterSet.Config as cms

ecalBarrelTimingTask = cms.EDAnalyzer("EBTimingTask",
    prefixME = cms.untracked.string('Ecal'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    useBeamStatus = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    L1GtEvmReadoutRecord = cms.InputTag("l1GtEvmUnpack"),
    energyThreshold = cms.untracked.double(1.0)
)

