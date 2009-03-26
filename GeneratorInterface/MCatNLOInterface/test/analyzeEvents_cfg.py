import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/home/xv/fabstoec/mcnloscratch/mcatnloTTee.root')
#                            fileNames = cms.untracked.vstring('file:/home/xv/fabstoec/mcnloscratch/mcatnloWWee.root')
#                            fileNames = cms.untracked.vstring('file:/home/xv/fabstoec/mcnloscratch/mcatnloZee.root')
)

process.myanalysis = cms.EDAnalyzer("WWeeAnalyzer",
                                    OutputFilename = cms.untracked.string('TTee_histos.root')
#                                    OutputFilename = cms.untracked.string('WWee_histos.root')
#                                    OutputFilename = cms.untracked.string('Zee_histos.root')
)

process.p = cms.Path(process.myanalysis)


