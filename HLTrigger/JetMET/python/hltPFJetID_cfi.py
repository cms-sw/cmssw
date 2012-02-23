import FWCore.ParameterSet.Config as cms

hltJetID = cms.EDProducer("HLTJetIDProducer",
    jetsInput = cms.InputTag("hltAntiKT5PFJets"),
    min_NHEF = cms.double(-999.),
    max_NHEF = cms.double(999.),
    min_NEMF = cms.double(-999.),
    max_NEMF = cms.double(999.),
    min_CEMF = cms.double(-999.),
    max_CEMF = cms.double(999.),
    min_CHEF = cms.double(-999.),
    max_CHEF = cms.double(999.),
    min_pt   = cms.double(30.)  
)


