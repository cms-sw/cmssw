import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFProducer.pfLinker_cfi
import RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi
particleFlowPrePuppi = RecoParticleFlow.PFProducer.pfLinker_cfi.pfLinker.clone()
particleFlowPrePuppi.PFCandidate = [cms.InputTag("particleFlowTmp")]
particleFlowPtrs = RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi.particleFlowTmpPtrs.clone()
particleFlowPtrs.src = "particleFlow"  # SRR 17-Dec-2019: 'particleFlow' will be post-puppi now. 

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(particleFlowPrePuppi, forceElectronsInHGCAL = True)
