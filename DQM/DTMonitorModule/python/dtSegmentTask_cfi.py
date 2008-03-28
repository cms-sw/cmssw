import FWCore.ParameterSet.Config as cms

dtSegmentAnalysisMonitor = cms.EDFilter("DTSegmentAnalysisTask",
    debug = cms.untracked.bool(False),
    recHits4DLabel = cms.string('dt4DSegments'),
    checkNoisyChannels = cms.untracked.bool(True),
    localrun = cms.untracked.bool(True),
    MTCC = cms.untracked.bool(True)
)


