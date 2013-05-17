import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.load('Configuration.EventContent.EventContent_cff')

process.source = cms.Source("PoolSource",
                            processingMode = cms.untracked.string('RunsAndLumis'),
                            fileNames = cms.untracked.vstring("file:dqm_file1_oldf.root",
                                                              "file:dqm_file3_oldf.root"))

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)
process.out = cms.OutputModule("PoolOutputModule",
                               splitLevel = cms.untracked.int32(0),
                               outputCommands = process.DQMEventContent.outputCommands,
                               fileName = cms.untracked.string('dqm_merged_file1_file3_oldf.root'),
                               dataset = cms.untracked.PSet(
                                 filterName = cms.untracked.string(''),
                                 dataTier = cms.untracked.string('')
                                 )
                               )

process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

