import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.tppfCandidatesOnPFCandidates_cfi import tppfCandidatesOnPFCandidates

pfNoPileUp = tppfCandidatesOnPFCandidates.clone(
    enable =  True,
    name = "pileUpOnPFCandidates",
    topCollection = "pfPileUp",
    bottomCollection = "particleFlowTmpPtrs",
    matchByPtrDirect = True
)
