import FWCore.ParameterSet.Config as cms

ecalBarrelDaqInfoTask = cms.EDAnalyzer("EBDaqInfoTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
