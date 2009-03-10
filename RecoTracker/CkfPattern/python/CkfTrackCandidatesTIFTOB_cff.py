import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# TrackerTrajectoryBuilders
#include "RecoTracker/CkfPattern/data/CkfTrajectoryBuilderESProducer.cff"
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducerTIFTOB_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesTIFTOB = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesTIFTOB.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesTIFTOB.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilderTIFTOB'
ckfTrackCandidatesTIFTOB.SeedProducer = 'combinatorialcosmicseedfinderTIFTOB'

