import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFProducer.pfLinker_cfi
particleFlow = RecoParticleFlow.PFProducer.pfLinker_cfi.pfLinker.clone()
particleFlow.PFCandidate = [cms.InputTag("particleFlowTmp")]
