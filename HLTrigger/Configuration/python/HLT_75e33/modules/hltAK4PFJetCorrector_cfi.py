import FWCore.ParameterSet.Config as cms

hltAK4PFJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltAK4PFJetCorrectorL1", "hltAK4PFJetCorrectorL2", "hltAK4PFJetCorrectorL3")
)
# foo bar baz
# LDP046it4Ffei
# w6mhJetMSohQU
