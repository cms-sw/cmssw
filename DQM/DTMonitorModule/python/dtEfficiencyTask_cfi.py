import FWCore.ParameterSet.Config as cms

dtEfficiencyMonitor = cms.EDAnalyzer("DTEfficiencyTask",
    # switch for verbosity
    debug = cms.untracked.bool(False),
    # labels of 4D and 1D hits
    recHits4DLabel = cms.string('dt4DSegments'),
    recHitLabel = cms.string('dt1DRecHits'),
    # interval of lumi block after which we reset the histos
    ResetCycle = cms.untracked.int32(10000)
)


