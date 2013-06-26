import FWCore.ParameterSet.Config as cms

hltPixlMBForAlignment = cms.EDFilter("HLTPixlMBForAlignmentFilter",
    pixlTag = cms.InputTag("hltPixelCands"),
    saveTags = cms.bool( False ),
    MinIsol = cms.double(0.05), ## minimum eta-phi isolation around tracks

    MinTrks = cms.uint32(2), ## Number of tracks required

    MinPt = cms.double(5.0), ## MinPt currently not used (all pt accepted)

    MinSep = cms.double(1.0) ## minimum eta-phi separation between tracks

)


