import FWCore.ParameterSet.Config as cms

hltRapGapFilter = cms.EDFilter("HLTRapGapFilter",
    saveTags = cms.bool( False ),
    maxEta = cms.double(5.0),
    minEta = cms.double(3.0),
    caloThresh = cms.double(20.0),
    inputTag = cms.InputTag("iterativeCone5CaloJets")
)


