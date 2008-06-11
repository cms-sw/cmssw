import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFTracking.vertexFilter_cfi
secStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
iterativeSecondTrackFiltering = cms.Sequence(secStep)
secStep.recTracks = cms.InputTag("iterativeSecondTrackMerging")
secStep.TrackAlgorithm = 'iter2'


