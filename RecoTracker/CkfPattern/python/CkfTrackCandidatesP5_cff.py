import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderP5_cff import *
# TrajectoryCleaning
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import *
# Navigation School
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesP5 = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesP5.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesP5.TrajectoryBuilderPSet.refToPSet_ = 'GroupedCkfTrajectoryBuilderP5'
#replace ckfTrackCandidatesP5.TrajectoryBuilder        = "CkfTrajectoryBuilderP5"
ckfTrackCandidatesP5.src = cms.InputTag('combinatorialcosmicseedfinderP5')

