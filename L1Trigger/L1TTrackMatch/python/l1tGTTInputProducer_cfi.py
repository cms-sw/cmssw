import FWCore.ParameterSet.Config as cms

l1tGTTInputProducer = cms.EDProducer('L1GTTInputProducer',
  l1TracksInputTag = cms.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"),
  outputCollectionName = cms.string("Level1TTTracksConverted"),
  setTrackWordBits = cms.bool(True),
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3
)

l1tGTTInputProducerExtended = cms.EDProducer('L1GTTInputProducer',
  l1TracksInputTag = cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),
  outputCollectionName = cms.string("Level1TTTracksExtendedConverted"),
  setTrackWordBits = cms.bool(True),
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3
)
