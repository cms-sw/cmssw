import FWCore.ParameterSet.Config as cms
import CommonTools.ParticleFlow.tppfJetsOnPFCandidates_cfi as _mod

pfNoJet = _mod.tppfJetsOnPFCandidates.clone(
    enable = True,
    name = "noJet",
    topCollection = "pfJetsPtrs",
    bottomCollection = "pfNoElectronJME",
)
# foo bar baz
# dI7bB78TGXtaL
