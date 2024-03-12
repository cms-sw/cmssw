import FWCore.ParameterSet.Config as cms
import CommonTools.ParticleFlow.tppfCandidatesOnPFCandidates_cfi as _mod

pfNoPileUp = _mod.tppfCandidatesOnPFCandidates.clone(
    enable =  True,
    name = "pileUpOnPFCandidates",
    topCollection = "pfPileUp",
    bottomCollection = "particleFlowTmpPtrs",
    matchByPtrDirect = True
)
# foo bar baz
# 9EqkCST4yBEVa
