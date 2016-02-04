import FWCore.ParameterSet.Config as cms

HLTgeneralTracks = cms.EDProducer("FastTrackMerger",
    # new quality setting
    newQuality = cms.untracked.string('confirmed'),
    # set new quality for confirmed tracks
    promoteTrackQuality = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(
       cms.InputTag("zeroStepFilter"),
       cms.InputTag("firstfilter"),
       cms.InputTag("secfilter"),
       cms.InputTag("thfilter"),
       cms.InputTag("foufilter"),
    )
)


