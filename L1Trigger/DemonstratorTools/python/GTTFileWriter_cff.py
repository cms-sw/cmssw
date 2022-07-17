import FWCore.ParameterSet.Config as cms

GTTFileWriter = cms.EDAnalyzer('GTTFileWriter',
  tracks = cms.untracked.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
  convertedTracks = cms.untracked.InputTag("L1GTTInputProducer","Level1TTTracksConverted"),
  vertices = cms.untracked.InputTag("VertexProducer", "l1verticesEmulation"),
  inputFilename = cms.untracked.string("L1GTTInputFile"),
  inputConvertedFilename = cms.untracked.string("L1GTTInputConvertedFile"),
  outputFilename = cms.untracked.string("L1GTTOutputToCorrelatorFile"),
  format = cms.untracked.string("APx")
)