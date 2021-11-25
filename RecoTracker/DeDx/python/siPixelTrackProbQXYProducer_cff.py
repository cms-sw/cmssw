import FWCore.ParameterSet.Config as cms

siPixelTrackProbQXY = cms.EDProducer("SiPixelTrackProbQXYProducer",
    tracks                     = cms.InputTag("generalTracks"),
)

 
doSiPixelTrackProbQXYTask = cms.Task(siPixelTrackProbQXY)
doSiPixelTrackProbQXY = cms.Sequence(doSiPixelTrackProbQXYTask)
