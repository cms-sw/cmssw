import FWCore.ParameterSet.Config as cms

generalTracks = cms.EDFilter("FastTrackMerger",
    # new quality setting
    newQuality = cms.untracked.string('confirmed'),
    # set new quality for confirmed tracks
    promoteTrackQuality = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(
       cms.InputTag("firstfilter"),
       cms.InputTag("secStep"),
       cms.InputTag("thStep"),
       cms.InputTag("fouStep"),
       cms.InputTag("fifthStep"),
    )
)


