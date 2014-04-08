import FWCore.ParameterSet.Config as cms

#special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagator_cff import *

from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cff import *
import  TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
ckfTrajectoryFilterBeamHaloMuon = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone()
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi import *
# clone the trajectory builder
CkfTrajectoryBuilderBeamHalo = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
beamhaloTrackCandidates = copy.deepcopy(ckfTrackCandidates)

ckfTrajectoryFilterBeamHaloMuon.minimumNumberOfHits = 4
ckfTrajectoryFilterBeamHaloMuon.minPt = 0.1
ckfTrajectoryFilterBeamHaloMuon.maxLostHits = 3
ckfTrajectoryFilterBeamHaloMuon.maxConsecLostHits = 2

CkfTrajectoryBuilderBeamHalo.propagatorAlong = 'BeamHaloPropagatorAlong'
CkfTrajectoryBuilderBeamHalo.propagatorOpposite = 'BeamHaloPropagatorOpposite'
CkfTrajectoryBuilderBeamHalo.trajectoryFilter.refToPSet_ = 'ckfTrajectoryFilterBeamHaloMuon'
beamhaloTrackCandidates.src = cms.InputTag('beamhaloTrackerSeeds')
beamhaloTrackCandidates.NavigationSchool = 'BeamHaloNavigationSchool'
beamhaloTrackCandidates.TransientInitialStateEstimatorParameters.propagatorAlongTISE = 'BeamHaloPropagatorAlong'
beamhaloTrackCandidates.TransientInitialStateEstimatorParameters.propagatorOppositeTISE = 'BeamHaloPropagatorOpposite'
beamhaloTrackCandidates.TrajectoryBuilderPSet.refToPSet_ = 'CkfTrajectoryBuilderBeamHalo'

