import FWCore.ParameterSet.Config as cms

hltMonoCaloJetFilter = cms.PSet(
    saveTags = cms.bool( False ),
    inputJetTag = cms.InputTag("iterativeCone5CaloJets"),
    max_PtSecondJet = cms.double(9999.0),
    max_DeltaPhi = cms.double(3.5)
)
