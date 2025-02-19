import FWCore.ParameterSet.Config as cms

lightpftrackprod = cms.EDProducer("LightPFTrackProducer",
    TrackQuality = cms.string('highPurity'),
    UseQuality = cms.bool(True),
    TkColList = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("secStep"), cms.InputTag("thStep"))
)


