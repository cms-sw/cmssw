import FWCore.ParameterSet.Config as cms

hltAK4PFCHSJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltAK4PFCHSJetCorrectorL1", "hltAK4PFCHSJetCorrectorL2", "hltAK4PFCHSJetCorrectorL3")
)
