import FWCore.ParameterSet.Config as cms

ecalBarrelTimingTask = cms.EDAnalyzer("EBTimingTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    useBeamStatus = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    L1GtEvmReadoutRecord = cms.InputTag("l1GtEvmUnpack")
)

