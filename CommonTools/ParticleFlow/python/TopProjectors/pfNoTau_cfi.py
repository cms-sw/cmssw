import FWCore.ParameterSet.Config as cms
import CommonTools.ParticleFlow.tppfTausOnPFJetsDeltaR_cfi as _mod

pfNoTau = _mod.tppfTausOnPFJetsDeltaR.clone(
    enable = True,
    deltaR = 0.5,
    name = "noTau",
    topCollection = "pfTausPtrs",
    bottomCollection = "pfJetsPtrs",
)
# foo bar baz
# zG7oVvA4kEvCb
# Ukj5yqxd2wqWX
