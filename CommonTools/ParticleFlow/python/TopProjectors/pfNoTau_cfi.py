import FWCore.ParameterSet.Config as cms
from CommonTools.ParticleFlow.tppfTausOnPFJetsDeltaR_cfi import tppfTausOnPFJetsDeltaR

pfNoTau = tppfTausOnPFJetsDeltaR.clone(
    enable = True,
    deltaR = 0.5,
    name = "noTau",
    topCollection = "pfTausPtrs",
    bottomCollection = "pfJetsPtrs",
)
