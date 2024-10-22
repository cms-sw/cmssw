import FWCore.ParameterSet.Config as cms

hltAK4PFPuppiJetsCorrected = cms.EDProducer("CorrectedPFJetProducer",
    correctors = cms.VInputTag("hltAK4PFPuppiJetCorrector"),
    src = cms.InputTag("hltAK4PFPuppiJets")
)
