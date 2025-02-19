import FWCore.ParameterSet.Config as cms

hltEgammaAllCombMassFilter = cms.EDFilter("HLTEgammaAllCombMassFilter",
    firstLegLastFilter = cms.InputTag("firstFilter"),
    secondLegLastFilter = cms.InputTag("secondFilter"),
    minMass = cms.double(-1.0),
    saveTags = cms.bool( False )
)


