import FWCore.ParameterSet.Config as cms

hltAK8PFJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltAK8PFJetCorrectorL1", "hltAK8PFJetCorrectorL2", "hltAK8PFJetCorrectorL3")
)
