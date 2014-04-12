import FWCore.ParameterSet.Config as cms

ecalBarrelTrendClient = cms.EDAnalyzer("EBTrendClient",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False)
)
