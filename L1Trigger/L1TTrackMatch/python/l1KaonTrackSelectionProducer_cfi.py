import FWCore.ParameterSet.Config as cms

l1KaonTrackSelectionProducer = cms.EDProducer('L1KaonTrackSelectionProducer',
  l1TracksInputTag = cms.InputTag("l1tTrackSelectionProducer","Level1TTTracksSelected"),                                            
  outputCollectionName = cms.string("Level1TTKaonTracksSelected"),
  processSimulatedTracks = cms.bool(True), # return selected tracks after cutting on the floating point values
  processEmulatedTracks = cms.bool(True), # return selected tracks after cutting on the bitwise emulated values
  debug = cms.int32(4) # Verbosity levels: 0, 1, 2, 3, 4
)

