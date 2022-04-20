import FWCore.ParameterSet.Config as cms

hltAK8PFCHSJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltAK8PFCHSJetCorrectorL1", "hltAK8PFCHSJetCorrectorL2", "hltAK8PFCHSJetCorrectorL3")
)
