import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFTracking.vertexFilter_cfi
thStep = RecoParticleFlow.PFTracking.vertexFilter_cfi.vertFilter.clone()
thStep.recTracks = cms.InputTag("thWithMaterialTracks")
thStep.DistZFromVertex = 0.1
thStep.TrackAlgorithm = 'iter3'

