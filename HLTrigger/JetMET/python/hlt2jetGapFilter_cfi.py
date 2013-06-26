import FWCore.ParameterSet.Config as cms

hlt2jetGapFilter = cms.EDFilter("HLT2jetGapFilter",
    inputTag = cms.InputTag("iterativeCone5CaloJets"),
    saveTags = cms.bool( False ),
    minEt = cms.double(90.0),
    minEta = cms.double(1.9)
)


