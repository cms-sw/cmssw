import FWCore.ParameterSet.Config as cms

ecalBarrelDcsInfoTask = cms.EDAnalyzer("EBDcsInfoTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False)
)
