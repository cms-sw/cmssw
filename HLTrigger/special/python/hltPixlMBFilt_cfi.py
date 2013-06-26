import FWCore.ParameterSet.Config as cms

hltPixlMBFilt = cms.EDFilter("HLTPixlMBFilt",
    pixlTag = cms.InputTag("hltPixelCands"),
    saveTags = cms.bool( False ),
    MinTrks = cms.uint32(2), ## Number of tracks from same vertex required

    MinPt = cms.double(0.0), ## MinPt currently not used (all pt accepted)

    MinSep = cms.double(1.0) ## minimum eta-phi separation between tracks

)


