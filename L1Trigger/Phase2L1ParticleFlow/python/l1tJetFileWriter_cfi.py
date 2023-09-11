import FWCore.ParameterSet.Config as cms

l1tSeededConeJetFileWriter = cms.EDAnalyzer('L1CTJetFileWriter',
  jets = cms.InputTag("l1tSCPFL1PuppiEmulatorCorrected"),
  nJets = cms.uint32(12),
  nFramesPerBX = cms.uint32(9), # 360 MHz clock or 25 Gb/s link
  TMUX = cms.uint32(6),
  maxLinesPerFile = cms.uint32(1024),
  outputFilename = cms.string("L1CTSCJetsPatterns"),
  format = cms.string("EMPv2"),
  outputFileExtension = cms.string("txt.gz")
)
