import FWCore.ParameterSet.Config as cms

# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# TrackerTrajectoryBuilders
#include "RecoTracker/CkfPattern/data/CkfTrajectoryBuilderESProducer.cff"
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducerTIFTOBTEC_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesTIFTOBTEC = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesTIFTOBTEC.NavigationSchool = 'CosmicNavigationSchool'
ckfTrackCandidatesTIFTOBTEC.TrajectoryBuilder = 'GroupedCkfTrajectoryBuilderTIFTOBTEC'
ckfTrackCandidatesTIFTOBTEC.SeedProducer = 'combinatorialcosmicseedfinderTIFTOBTEC'

