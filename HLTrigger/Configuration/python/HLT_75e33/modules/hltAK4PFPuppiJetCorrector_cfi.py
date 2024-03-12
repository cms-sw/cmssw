import FWCore.ParameterSet.Config as cms

hltAK4PFPuppiJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltAK4PFPuppiJetCorrectorL1", "hltAK4PFPuppiJetCorrectorL2", "hltAK4PFPuppiJetCorrectorL3")
)
# foo bar baz
# M2JjPRCP92MLE
