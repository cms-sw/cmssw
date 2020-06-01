import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.tppfJetsOnPFCandidates_cfi import tppfJetsOnPFCandidates

pfNoJet = tppfJetsOnPFCandidates.clone(
    enable = True,
    name = "noJet",
    topCollection = "pfJetsPtrs",
    bottomCollection = "pfNoElectronJME",
)
