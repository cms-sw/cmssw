import FWCore.ParameterSet.Config as cms

dtSegmentAnalysisMonitor = cms.EDFilter("DTSegmentAnalysisTask",
    # switch for verbosity
    debug = cms.untracked.bool(False),
    # label of 4D segments
    recHits4DLabel = cms.string('dt4DSegments'),
    # skip segments with noisy cells (reads from DB)
    checkNoisyChannels = cms.untracked.bool(True),
    # switch for local-mode runs
    localrun = cms.untracked.bool(True)
)


