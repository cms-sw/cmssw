import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
iterativeThirdTracksWithPairs = copy.deepcopy(ctfWithMaterialTracks)
iterativeThirdTracks = cms.Sequence(iterativeThirdTracksWithPairs)
iterativeThirdTracksWithPairs.src = 'iterativeThirdTrackCandidatesWithPairs'
iterativeThirdTracksWithPairs.TTRHBuilder = 'WithoutRefit'
iterativeThirdTracksWithPairs.Fitter = 'KFFittingSmoother'
iterativeThirdTracksWithPairs.Propagator = 'PropagatorWithMaterial'

