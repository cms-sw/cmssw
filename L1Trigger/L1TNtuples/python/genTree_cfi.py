import FWCore.ParameterSet.Config as cms

genTree = cms.EDAnalyzer(
    "L1GenTreeProducer",
    genJetToken     = cms.untracked.InputTag("ak4GenJetsNoNu"),
    genMETTrueToken = cms.untracked.InputTag("genMetTrue"),
    genMETCaloToken     = cms.untracked.InputTag("genMetCalo"),
    genParticleToken = cms.untracked.InputTag("genParticles"),
    pileupInfoToken     = cms.untracked.InputTag("addPileupInfo")
)
