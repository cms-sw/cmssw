import FWCore.ParameterSet.Config as cms

## VarParsing
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.setDefault('maxEvents', 1)
options.parseArguments()

process = cms.Process('TEST')

process.options.numberOfThreads = 1
process.options.numberOfStreams = 0
process.options.wantSummary = False
process.maxEvents.input = options.maxEvents

## Source
process.source = cms.Source('PoolSource',
  fileNames = cms.untracked.vstring(options.inputFiles),
  inputCommands = cms.untracked.vstring('drop *')
)
