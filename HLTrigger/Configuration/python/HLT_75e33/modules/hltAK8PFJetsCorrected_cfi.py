import FWCore.ParameterSet.Config as cms

hltAK8PFJetsCorrected = cms.EDProducer("CorrectedPFJetProducer",
    correctors = cms.VInputTag("hltAK8PFJetCorrector"),
    src = cms.InputTag("hltAK8PFJets")
)
