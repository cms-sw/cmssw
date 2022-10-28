import FWCore.ParameterSet.Config as cms

hltAK8PFPuppiJetsCorrected = cms.EDProducer("CorrectedPFJetProducer",
    correctors = cms.VInputTag("hltAK8PFPuppiJetCorrector"),
    src = cms.InputTag("hltAK8PFPuppiJets")
)
