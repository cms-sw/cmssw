import FWCore.ParameterSet.Config as cms

L1GTTInputProducer = cms.EDProducer('L1GTTInputProducer',
  l1TracksInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
  outputCollectionName = cms.string("Level1TTTracksConverted"),
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3
)

L1GTTInputProducerExtended = cms.EDProducer('L1GTTInputProducer',
  l1TracksInputTag = cms.InputTag("TTTracksFromExtendedTrackletEmulation", "Level1TTTracks"),
  outputCollectionName = cms.string("Level1TTTracksExtendedConverted"),
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3
)
