import FWCore.ParameterSet.Config as cms

ecalBarrelDaqInfoTask = cms.EDAnalyzer("EBDaqInfoTask",
    prefixME = cms.untracked.string('Ecal'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
