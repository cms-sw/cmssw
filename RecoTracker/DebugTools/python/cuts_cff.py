import FWCore.ParameterSet.Config as cms

import PhysicsTools.RecoAlgos.recoTrackSelector_cfi
cutsRTAlgoA = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()
cutsRTAlgoB = PhysicsTools.RecoAlgos.recoTrackSelector_cfi.recoTrackSelector.clone()

import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
cutsTPEffic = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
cutsTPFake  = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
