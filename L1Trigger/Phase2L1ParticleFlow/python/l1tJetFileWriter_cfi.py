import FWCore.ParameterSet.Config as cms

l1tSeededConeJetFileWriter = cms.EDAnalyzer('L1CTJetFileWriter',
  collections = cms.VPSet(cms.PSet(jets = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulator"),
                                   nJets = cms.uint32(12),
                                   mht  = cms.InputTag("l1tSC4PFL1PuppiCorrectedEmulatorMHT"),
                                   nSums = cms.uint32(1))),
  nFramesPerBX = cms.uint32(9), # 360 MHz clock or 25 Gb/s link
  TMUX = cms.uint32(6),
  maxLinesPerFile = cms.uint32(1024),
  outputFilename = cms.string("L1CTSCJetsPatterns"),
  format = cms.string("EMPv2"),
  outputFileExtension = cms.string("txt.gz")
)
