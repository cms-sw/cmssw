import FWCore.ParameterSet.Config as cms

import RecoParticleFlow.PFProducer.pfLinker_cfi
import RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi
particleFlow = RecoParticleFlow.PFProducer.pfLinker_cfi.pfLinker.clone(
    PFCandidate = ["particleFlowTmp"]
)
particleFlowPtrs = RecoParticleFlow.PFProducer.particleFlowTmpPtrs_cfi.particleFlowTmpPtrs.clone(
    src = "particleFlow"
)
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify(particleFlow, forceElectronsInHGCAL = True)
