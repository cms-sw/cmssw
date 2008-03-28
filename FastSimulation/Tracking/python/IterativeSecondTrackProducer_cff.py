import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
iterativeSecondTracksWithTriplets = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
iterativeSecondTracksWithPairs = copy.deepcopy(ctfWithMaterialTracks)
iterativeSecondTracks = cms.Sequence(iterativeSecondTracksWithTriplets+iterativeSecondTracksWithPairs)
iterativeSecondTracksWithTriplets.src = 'iterativeSecondTrackCandidatesWithTriplets'
iterativeSecondTracksWithTriplets.TTRHBuilder = 'WithoutRefit'
iterativeSecondTracksWithTriplets.Fitter = 'KFFittingSmoother'
iterativeSecondTracksWithTriplets.Propagator = 'PropagatorWithMaterial'
iterativeSecondTracksWithPairs.src = 'iterativeSecondTrackCandidatesWithPairs'
iterativeSecondTracksWithPairs.TTRHBuilder = 'WithoutRefit'
iterativeSecondTracksWithPairs.Fitter = 'KFFittingSmoother'
iterativeSecondTracksWithPairs.Propagator = 'PropagatorWithMaterial'

