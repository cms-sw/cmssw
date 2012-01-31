import FWCore.ParameterSet.Config as cms

hltMonoPFJetFilter = cms.PSet(
    saveTags = cms.bool( False ),
    inputJetTag = cms.InputTag("hltAntiKT5PFJets"),
    max_PtSecondJet = cms.double(9999.0),
    max_DeltaPhi = cms.double(3.5)
)
