import FWCore.ParameterSet.Config as cms

hltAK4PFJetsCorrected = cms.EDProducer("CorrectedPFJetProducer",
    correctors = cms.VInputTag("hltAK4PFJetCorrector"),
    src = cms.InputTag("hltAK4PFJets")
)
