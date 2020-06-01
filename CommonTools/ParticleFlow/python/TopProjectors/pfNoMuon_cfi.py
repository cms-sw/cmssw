import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.tppfCandidatesOnPFCandidates_cfi import tppfCandidatesOnPFCandidates

pfNoMuon = tppfCandidatesOnPFCandidates.clone(
    enable = True,
    name = "noMuon",
    topCollection = "pfIsolatedMuons",
    bottomCollection = "pfNoPileUp",
)

pfNoMuonJME = pfNoMuon.clone(
    bottomCollection = "pfNoPileUpJME"
)
