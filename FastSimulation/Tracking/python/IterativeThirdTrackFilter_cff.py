import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFTracking.vertexFilter_cfi
thStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
iterativeThirdTrackFiltering = cms.Sequence(thStep)
thStep.recTracks = cms.InputTag("iterativeThirdTrackMerging")
thStep.TrackAlgorithm = 'iter3'
thStep.DistZFromVertex = 0.1


