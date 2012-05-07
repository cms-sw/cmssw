import FWCore.ParameterSet.Config as cms

hltJetID = cms.EDProducer("HLTJetIDProducer",
    jetsInput = cms.InputTag("hltMCJetCorJetIcone5HF07"),
    min_EMF = cms.double(0.0001),
    max_EMF = cms.double(999.),
    min_N90 = cms.int32(1)
)


