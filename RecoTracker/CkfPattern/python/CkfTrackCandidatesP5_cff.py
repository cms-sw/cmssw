import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmoother_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmoother_cfi import *
# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff import *
# TrajectoryCleaning
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import *
# Navigation School
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesP5 = ckfTrackCandidates.clone(
    NavigationSchool      = 'CosmicNavigationSchool',
    TrajectoryBuilderPSet = dict(refToPSet_ = 'GroupedCkfTrajectoryBuilderP5'),
    #replace ckfTrackCandidatesP5.TrajectoryBuilder        = "CkfTrajectoryBuilderP5"
    src                   = 'combinatorialcosmicseedfinderP5'
)
