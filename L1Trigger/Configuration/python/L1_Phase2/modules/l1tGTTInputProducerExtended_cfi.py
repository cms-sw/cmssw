import FWCore.ParameterSet.Config as cms

l1tGTTInputProducerExtended = cms.EDProducer("L1GTTInputProducer",
    debug = cms.int32(0),
    l1TracksInputTag = cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation","Level1TTTracks"),
    outputCollectionName = cms.string('Level1TTTracksExtendedConverted'),
    setTrackWordBits = cms.bool(True)
)
