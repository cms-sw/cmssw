import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# TrackerTrajectoryBuilders
#include "RecoTracker/CkfPattern/data/CkfTrajectoryBuilderESProducer.cff"
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducerTIFTIBTOB_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesTIFTIBTOB = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesTIFTIBTOB.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesTIFTIBTOB.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilderTIFTIBTOB'
ckfTrackCandidatesTIFTIBTOB.SeedProducer = 'combinatorialcosmicseedfinderTIFTIBTOB'

