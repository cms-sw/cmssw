import FWCore.ParameterSet.Config as cms

hltAK8PFCHSJetsCorrected = cms.EDProducer("CorrectedPFJetProducer",
    correctors = cms.VInputTag("hltAK8PFCHSJetCorrector"),
    src = cms.InputTag("hltAK8PFCHSJets")
)
