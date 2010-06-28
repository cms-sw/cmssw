import FWCore.ParameterSet.Config as cms

#special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagator_cff import *

from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cff import *
import  TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfTrajectoryFilterBeamHaloMuon = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
#include "RecoTracker/CkfPattern/data/GroupedCkfTrajectoryBuilderESProducer.cff"
# clone the trajectory builder
CkfTrajectoryBuilderBeamHalo = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
beamhaloTrackCandidates = copy.deepcopy(ckfTrackCandidates)

ckfTrajectoryFilterBeamHaloMuon.ComponentName = 'ckfTrajectoryFilterBeamHaloMuon'
ckfTrajectoryFilterBeamHaloMuon.filterPset.minimumNumberOfHits = 4
ckfTrajectoryFilterBeamHaloMuon.filterPset.minPt = 0.1
ckfTrajectoryFilterBeamHaloMuon.filterPset.maxLostHits = 3
ckfTrajectoryFilterBeamHaloMuon.filterPset.maxConsecLostHits = 2

CkfTrajectoryBuilderBeamHalo.ComponentName = 'CkfTrajectoryBuilderBH'
CkfTrajectoryBuilderBeamHalo.propagatorAlong = 'BeamHaloPropagatorAlong'
CkfTrajectoryBuilderBeamHalo.propagatorOpposite = 'BeamHaloPropagatorOpposite'
CkfTrajectoryBuilderBeamHalo.trajectoryFilterName = 'ckfTrajectoryFilterBeamHaloMuon'
beamhaloTrackCandidates.src = cms.InputTag('beamhaloTrackerSeeds')
beamhaloTrackCandidates.NavigationSchool = 'BeamHaloNavigationSchool'
beamhaloTrackCandidates.TransientInitialStateEstimatorParameters.propagatorAlongTISE = 'BeamHaloPropagatorAlong'
beamhaloTrackCandidates.TransientInitialStateEstimatorParameters.propagatorOppositeTISE = 'BeamHaloPropagatorOpposite'
beamhaloTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilderBH'

