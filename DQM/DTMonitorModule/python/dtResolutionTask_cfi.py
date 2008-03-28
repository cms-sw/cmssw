import FWCore.ParameterSet.Config as cms

dtResolutionAnalysisMonitor = cms.EDFilter("DTResolutionAnalysisTask",
    debug = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    recHitLabel = cms.string('dt1DRecHits'),
    MTCC = cms.untracked.bool(True)
)


