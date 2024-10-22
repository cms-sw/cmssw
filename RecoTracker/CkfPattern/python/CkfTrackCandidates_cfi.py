import FWCore.ParameterSet.Config as cms

import RecoTracker.CkfPattern.ckfTrackCandidateMaker_cfi as _mod
ckfTrackCandidates = _mod.ckfTrackCandidateMaker.clone(
    TrajectoryBuilderPSet = dict(refToPSet_ = cms.string('GroupedCkfTrajectoryBuilder')),
    maxSeedsBeforeCleaning = 5000,
)

ckfTrackCandidatesIterativeDefault = ckfTrackCandidates.clone(
    TrajectoryBuilderPSet = dict(refToPSet_ = cms.string('GroupedCkfTrajectoryBuilderIterativeDefault')),
)
