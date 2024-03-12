import FWCore.ParameterSet.Config as cms

hltPFMETJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltPFMETJetCorrectorL1", "hltPFMETJetCorrectorL2", "hltPFMETJetCorrectorL3")
)
# foo bar baz
# bsvqOYx1df8zC
# FJh9MGGr4yPDe
