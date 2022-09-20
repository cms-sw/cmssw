import FWCore.ParameterSet.Config as cms

l1tGTTInputProducer = cms.EDProducer('L1GTTInputProducer',
  l1TracksInputTag = cms.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"),
  outputCollectionName = cms.string("Level1TTTracksConverted"),
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3
)

l1tGTTInputProducerExtended = cms.EDProducer('L1GTTInputProducer',
  l1TracksInputTag = cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),
  outputCollectionName = cms.string("Level1TTTracksExtendedConverted"),
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3
)
