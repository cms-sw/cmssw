import FWCore.ParameterSet.Config as cms

GTTFileWriter = cms.EDAnalyzer('GTTFileWriter',
  tracks = cms.untracked.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"),
  convertedTracks = cms.untracked.InputTag("l1tGTTInputProducer","Level1TTTracksConverted"),
  vertices = cms.untracked.InputTag("l1tVertexProducer", "l1verticesEmulation"),
  inputFilename = cms.untracked.string("L1GTTInputFile"),
  inputConvertedFilename = cms.untracked.string("L1GTTInputConvertedFile"),
  outputFilename = cms.untracked.string("L1GTTOutputToCorrelatorFile"),
  format = cms.untracked.string("APx")
)
