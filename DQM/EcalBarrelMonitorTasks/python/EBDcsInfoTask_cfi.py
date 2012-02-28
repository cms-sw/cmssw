import FWCore.ParameterSet.Config as cms

ecalBarrelDcsInfoTask = cms.EDAnalyzer("EBDcsInfoTask",
    prefixME = cms.untracked.string('Ecal'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
