import FWCore.ParameterSet.Config as cms

l1tGTTFileReader = cms.EDProducer('GTTFileReader',
  files = cms.vstring("L1GTTOutputToCorrelatorFile_0.txt"),
  filesInputTracks = cms.vstring("L1GTTInputFile_0.txt"),
  l1VertexCollectionName = cms.string("L1VerticesFirmware"),
  l1TrackCollectionName = cms.string("Level1TTTracks"),
  format = cms.untracked.string("APx")
)
