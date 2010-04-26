import FWCore.ParameterSet.Config as cms

ecalEndcapTrendClient = cms.EDAnalyzer("EETrendClient",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False)
)
