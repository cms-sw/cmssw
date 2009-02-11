import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# TrackerTrajectoryBuilders
#include "RecoTracker/CkfPattern/data/CkfTrajectoryBuilderESProducer.cff"
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducerTIFTIB_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesTIFTIB = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesTIFTIB.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesTIFTIB.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilderTIFTIB'
ckfTrackCandidatesTIFTIB.SeedProducer = 'combinatorialcosmicseedfinderTIFTIB'

