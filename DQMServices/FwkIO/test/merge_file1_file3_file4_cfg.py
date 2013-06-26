import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:dqm_file1.root","file:dqm_file3.root","file:dqm_file4.root"))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_merged_file1_file3_file4.root"))
process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

