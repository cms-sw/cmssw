import FWCore.ParameterSet.Config as cms

process = cms.Process("CopyFile")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(100))
readFiles = cms.untracked.vstring('/store/data/Commissioning10/MinimumBias/RAW/v4/000/132/440/049F45A5-F53B-DF11-A9FC-0030487A3DE0.root')
secFiles = cms.untracked.vstring()
process.source = cms.Source('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)


process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring("keep *"),
                                  fileName = cms.untracked.string('Raw.root')
                                  )

process.out_step = cms.EndPath(process.output)


