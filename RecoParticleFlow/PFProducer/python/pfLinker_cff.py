import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFProducer.pfLinker_cfi
import RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi
particleFlow = RecoParticleFlow.PFProducer.pfLinker_cfi.pfLinker.clone()
particleFlow.PFCandidate = [cms.InputTag("particleFlowTmp")]
particleFlowPtrs = RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi.particleFlowTmpPtrs.clone()
particleFlowPtrs.src = "particleFlow"
