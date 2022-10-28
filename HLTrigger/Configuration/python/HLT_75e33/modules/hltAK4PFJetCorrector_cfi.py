import FWCore.ParameterSet.Config as cms

hltAK4PFJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltAK4PFJetCorrectorL1", "hltAK4PFJetCorrectorL2", "hltAK4PFJetCorrectorL3")
)
