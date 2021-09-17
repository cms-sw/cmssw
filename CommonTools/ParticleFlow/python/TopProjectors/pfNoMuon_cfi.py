import FWCore.ParameterSet.Config as cms
import CommonTools.ParticleFlow.tppfCandidatesOnPFCandidates_cfi as _mod

pfNoMuon = _mod.tppfCandidatesOnPFCandidates.clone(
    enable = True,
    name = "noMuon",
    topCollection = "pfIsolatedMuons",
    bottomCollection = "pfNoPileUp",
)

pfNoMuonJME = pfNoMuon.clone(
    bottomCollection = "pfNoPileUpJME"
)
