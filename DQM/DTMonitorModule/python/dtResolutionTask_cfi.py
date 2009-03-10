import FWCore.ParameterSet.Config as cms

dtResolutionAnalysisMonitor = cms.EDAnalyzer("DTResolutionAnalysisTask",
    # labels of 4D and 1D hits
    recHits4DLabel = cms.string('dt4DSegments'),
    recHitLabel = cms.string('dt1DRecHits'),
    # interval of lumi block after which we reset the histos
    ResetCycle = cms.untracked.int32(10000)
)


