import FWCore.ParameterSet.Config as cms

#special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagator_cff import *
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cff import *
import  TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *

ckfTrajectoryFilterBeamHaloMuon = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 4,
    minPt               = 0.1,
    maxLostHits         = 3,
    maxConsecLostHits   = 2
)

# clone the trajectory builder
CkfTrajectoryBuilderBeamHalo = CkfTrajectoryBuilder.clone(
    propagatorAlong = 'BeamHaloPropagatorAlong',
    propagatorOpposite = 'BeamHaloPropagatorOpposite',
    trajectoryFilter = dict(refToPSet_ = 'ckfTrajectoryFilterBeamHaloMuon')
)

# generate CTF track candidates ############
beamhaloTrackCandidates = ckfTrackCandidates.clone(
   src = 'beamhaloTrackerSeeds',
   NavigationSchool = 'BeamHaloNavigationSchool',
   TransientInitialStateEstimatorParameters = dict(
	propagatorAlongTISE = 'BeamHaloPropagatorAlong',
	propagatorOppositeTISE = 'BeamHaloPropagatorOpposite'),
   TrajectoryBuilderPSet = dict(refToPSet_ = 'CkfTrajectoryBuilderBeamHalo')
)
