import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# TrackerTrajectoryBuilders
#include "RecoTracker/CkfPattern/data/CkfTrajectoryBuilderESProducerTIF.cff"
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducerTIF_cff import *
# TrajectoryCleaning
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import *
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesTIF = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesTIF.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesTIF.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilderTIF'
#replace ckfTrackCandidatesTIF.TrajectoryBuilder        = "CkfTrajectoryBuilderTIF"
ckfTrackCandidatesTIF.SeedProducer = 'combinatorialcosmicseedfinderTIF'

