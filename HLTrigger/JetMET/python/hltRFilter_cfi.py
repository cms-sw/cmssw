import FWCore.ParameterSet.Config as cms

hltRFilter = cms.EDFilter( "HLTRFilter",
    inputTag = cms.InputTag("hltRHemisphere"),
    inputMetTag = cms.InputTag("hltMet"),
    minR = cms.double(0.3),
    minMR = cms.double(100.0),
    doRPrime = cms.bool( False),
    acceptNJ = cms.bool(True)
)
