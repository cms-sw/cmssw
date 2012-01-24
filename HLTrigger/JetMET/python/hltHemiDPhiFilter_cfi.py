import FWCore.ParameterSet.Config as cms

hltHemiDPhiFilter = cms.EDFilter("HLTHemiDPhiFilter",
    inputTag = cms.InputTag("hltRHemisphere"),
    saveTags = cms.bool( False ),
    minDphi = cms.double(2.9415),
    acceptNJ = cms.bool(True)
)


