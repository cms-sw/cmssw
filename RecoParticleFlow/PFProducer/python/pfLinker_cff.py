import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFProducer.pfLinker_cfi
import RecoParticleFlow.PFProducer.particleFlowPtrs_cfi
particleFlow = RecoParticleFlow.PFProducer.pfLinker_cfi.pfLinker.clone()
particleFlowPtrs = RecoParticleFlow.PFProducer.particleFlowPtrs_cfi.particleFlowTmpPtrs.clone()
particleFlow.PFCandidate = [cms.InputTag("particleFlowTmp")]
particleFlowPtrs.src = "particleFlow"
