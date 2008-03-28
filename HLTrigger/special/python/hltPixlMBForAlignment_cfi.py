import FWCore.ParameterSet.Config as cms

hltPixlMBForAlignment = cms.EDFilter("HLTPixlMBForAlignmentFilter",
    pixlTag = cms.InputTag("hltPixelCands"),
    MinIsol = cms.double(0.05),
    MinTrks = cms.uint32(2),
    MinPt = cms.double(5.0),
    MinSep = cms.double(1.0)
)


