import FWCore.ParameterSet.Config as cms

l1GeneratorTree = cms.EDAnalyzer(
    "L1EventTreeProducer",
    genJetToken     = cms.untracked.InputTag("ak4GenJets"),
    geParticleToken = cms.untracked.InputTag("genParticles"),
    puInfoToken     = cms.untracked.InputTag("addPileupInfo")
)
