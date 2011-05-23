import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFProducer.pfEgammaLinker_cfi
particleFlow = RecoParticleFlow.PFProducer.pfEgammaLinker_cfi.egammaLinker.clone()
particleFlow.PFCandidate = cms.InputTag("particleFlowTmp")
