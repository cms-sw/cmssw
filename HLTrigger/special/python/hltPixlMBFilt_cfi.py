import FWCore.ParameterSet.Config as cms

hltPixlMBFilt = cms.EDFilter("HLTPixlMBFilt",
    pixlTag = cms.InputTag("hltPixelCands"),
    MinTrks = cms.uint32(2),
    MinPt = cms.double(0.0),
    MinSep = cms.double(1.0)
)


