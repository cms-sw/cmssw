import FWCore.ParameterSet.Config as cms

GTTFileWriter = cms.EDAnalyzer('GTTFileWriter',
  tracks = cms.untracked.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"),
  convertedTracks = cms.untracked.InputTag("l1tGTTInputProducer", "Level1TTTracksConverted"),
  vertices = cms.untracked.InputTag("l1tVertexProducer", "l1verticesEmulation"),
  jets = cms.untracked.InputTag("l1tTrackJetsEmulation","L1TrackJets"),
  htmiss = cms.untracked.InputTag("l1tTrackerEmuHTMiss", "L1TrackerEmuHTMiss"),
  etmiss = cms.untracked.InputTag("l1tTrackerEmuEtMiss", "L1TrackerEmuEtMiss"),
  inputFilename = cms.untracked.string("L1GTTInputFile"),
  inputConvertedFilename = cms.untracked.string("L1GTTInputConvertedFile"),
  outputCorrelatorFilename = cms.untracked.string("L1GTTOutputToCorrelatorFile"),
  outputGlobalTriggerFilename = cms.untracked.string("L1GTTOutputToGlobalTriggerFile"),
  format = cms.untracked.string("APx")
)
