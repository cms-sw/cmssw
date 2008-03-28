import FWCore.ParameterSet.Config as cms

#special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagator_cff import *
# KFTrajectoryFitterESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectoryFitterESProducer.cfi"
#es_module KFTrajectoryFitterBeamHalo = KFTrajectoryFitter from "TrackingTools/TrackFitters/data/KFTrajectoryFitterESProducer.cfi"
#replace KFTrajectoryFitterBeamHalo.ComponentName = "KFFitterBH"
#replace KFTrajectoryFitterBeamHalo.Propagator = "BeamHaloPropagatorAlong"
# KFTrajectorySmootherESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectorySmootherESProducer.cfi"
#es_module KFTrajectorySmootherBeamHalo = KFTrajectorySmoother from "TrackingTools/TrackFitters/data/KFTrajectorySmootherESProducer.cfi"
#replace KFTrajectorySmootherBeamHalo.ComponentName = "KFSmootherBH"
#replace KFTrajectorySmootherBeamHalo.Propagator = "BeamHaloPropagatorAlong"
# KFFittingSmootherESProducer
#include "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"
#es_module KFFittingSmootherBeamHalo = KFFittingSmoother from "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"
#replace KFFittingSmootherBeamHalo.ComponentName = "KFFittingSmootherBH"
#replace KFFittingSmootherBeamHalo.Fitter = "KFFitterBH"
#replace KFFittingSmootherBeamHalo.Smoother = "KFSmootherBH"
# TrackerTrajectoryBuilders
# to get the dependencies
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cff import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
ckfTrajectoryFilterBeamHaloMuon = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi import *
#include "RecoTracker/CkfPattern/data/GroupedCkfTrajectoryBuilderESProducer.cff"
# clone the trajectory builder
CkfTrajectoryBuilderBeamHalo = copy.deepcopy(CkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesBeamHaloMuon = copy.deepcopy(ckfTrackCandidates)
ckfTrajectoryFilterBeamHaloMuon.ComponentName = 'ckfTrajectoryFilterBeamHaloMuon'
ckfTrajectoryFilterBeamHaloMuon.filterPset.minimumNumberOfHits = 4
#es_module CkfTrajectoryBuilderBeamHalo = GroupedCkfTrajectoryBuilder from "RecoTracker/CkfPattern/data/GroupedCkfTrajectoryBuilderESProducer.cfi"
CkfTrajectoryBuilderBeamHalo.ComponentName = 'CkfTrajectoryBuilderBH'
CkfTrajectoryBuilderBeamHalo.propagatorAlong = 'BeamHaloPropagatorAlong'
CkfTrajectoryBuilderBeamHalo.propagatorOpposite = 'BeamHaloPropagatorOpposite'
CkfTrajectoryBuilderBeamHalo.trajectoryFilterName = 'ckfTrajectoryFilterBeamHaloMuon'
ckfTrackCandidatesBeamHaloMuon.SeedProducer = 'combinatorialbeamhaloseedfinder'
ckfTrackCandidatesBeamHaloMuon.NavigationSchool = 'BeamHaloNavigationSchool'
ckfTrackCandidatesBeamHaloMuon.TransientInitialStateEstimatorParameters.propagatorAlongTISE = 'BeamHaloPropagatorAlong'
ckfTrackCandidatesBeamHaloMuon.TransientInitialStateEstimatorParameters.propagatorOppositeTISE = 'BeamHaloPropagatorOpposite'
ckfTrackCandidatesBeamHaloMuon.TrajectoryBuilder = 'CkfTrajectoryBuilderBH'

