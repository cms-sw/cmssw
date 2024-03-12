import FWCore.ParameterSet.Config as cms

lightpftrackprod = cms.EDProducer("LightPFTrackProducer",
    TrackQuality = cms.string('highPurity'),
    UseQuality = cms.bool(True),
    TkColList = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("secStep"), cms.InputTag("thStep"))
)


# foo bar baz
# 9M07mbRFsdxQW
# E43p7e21maT0N
