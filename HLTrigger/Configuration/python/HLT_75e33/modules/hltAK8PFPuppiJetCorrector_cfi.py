import FWCore.ParameterSet.Config as cms

hltAK8PFPuppiJetCorrector = cms.EDProducer("ChainedJetCorrectorProducer",
    correctors = cms.VInputTag("hltAK8PFPuppiJetCorrectorL1", "hltAK8PFPuppiJetCorrectorL2", "hltAK8PFPuppiJetCorrectorL3")
)
