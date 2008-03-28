import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
iterativeFirstTracksWithTriplets = copy.deepcopy(ctfWithMaterialTracks)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
iterativeFirstTracksWithPairs = copy.deepcopy(ctfWithMaterialTracks)
iterativeFirstTracks = cms.Sequence(iterativeFirstTracksWithTriplets+iterativeFirstTracksWithPairs)
iterativeFirstTracksWithTriplets.src = 'iterativeFirstTrackCandidatesWithTriplets'
iterativeFirstTracksWithTriplets.TTRHBuilder = 'WithoutRefit'
iterativeFirstTracksWithTriplets.Fitter = 'KFFittingSmoother'
iterativeFirstTracksWithTriplets.Propagator = 'PropagatorWithMaterial'
iterativeFirstTracksWithPairs.src = 'iterativeFirstTrackCandidatesWithPairs'
iterativeFirstTracksWithPairs.TTRHBuilder = 'WithoutRefit'
iterativeFirstTracksWithPairs.Fitter = 'KFFittingSmoother'
iterativeFirstTracksWithPairs.Propagator = 'PropagatorWithMaterial'

