import FWCore.ParameterSet.Config as cms

l1tGTTInputProducer = cms.EDProducer("L1GTTInputProducer",
    debug = cms.int32(0),
    l1TracksInputTag = cms.InputTag("l1tTTTracksFromTrackletEmulation","Level1TTTracks"),
    outputCollectionName = cms.string('Level1TTTracksConverted'),
    setTrackWordBits = cms.bool(True)
)
