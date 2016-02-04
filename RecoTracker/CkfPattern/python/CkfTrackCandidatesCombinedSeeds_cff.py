import FWCore.ParameterSet.Config as cms

# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cff import *
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cff import *
# TrajectoryCleaning
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import *
# navigation school
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesCombinedSeeds = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesCombinedSeeds.src = cms.InputTag('globalCombinedSeeds')

