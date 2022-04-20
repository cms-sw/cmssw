import FWCore.ParameterSet.Config as cms

hltAK4PFCHSJetsCorrected = cms.EDProducer("CorrectedPFJetProducer",
    correctors = cms.VInputTag("hltAK4PFCHSJetCorrector"),
    src = cms.InputTag("hltAK4PFCHSJets")
)
