import FWCore.ParameterSet.Config as cms

l1tGTTFileReader = cms.EDProducer('GTTFileReader',
  processOutputToCorrelator = cms.bool(True),
  processInputTracks = cms.bool(False),
  processOutputToGlobalTrigger = cms.bool(False), #Not fully implemented yet, partial skeleton added
  kEmptyFramesOutputToCorrelator = cms.untracked.uint32(0),
  kEmptyFramesInputTracks = cms.untracked.uint32(0),
  kEmptyFramesOutputToGlobalTrigger = cms.untracked.uint32(0),
  filesOutputToCorrelator = cms.vstring("L1GTTOutputToCorrelatorFile_0.txt"),
  filesInputTracks = cms.vstring("L1GTTInputFile_0.txt"),
  filesOutputToGlobalTrigger = cms.vstring("L1GTTOutputToGlobalTriggerFile_0.txt"),
  l1VertexCollectionName = cms.string("L1VerticesFirmware"),
  l1TrackCollectionName = cms.string("Level1TTTracks"),
  format = cms.untracked.string("APx")
)
