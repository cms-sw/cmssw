import FWCore.ParameterSet.Config as cms

lostTracks = cms.EDProducer("PATLostTracks",
    inputCandidates = cms.InputTag("particleFlow"),
    inputTracks = cms.InputTag("generalTracks"),
    minPt = cms.double(0.95),	
    minHits = cms.uint32(8),	
    minPixelHits = cms.uint32(1),	
)
