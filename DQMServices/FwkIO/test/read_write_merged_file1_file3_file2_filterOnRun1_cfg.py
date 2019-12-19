import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            reScope = cms.untracked.string(""),
                            fileNames = cms.untracked.vstring("file:dqm_merged_file1_file3_file2_filterOnRun1.root"))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_merged_file1_file3_file2_filterOnRun1_copy.root"))


process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore", forceResetOnBeginLumi = cms.untracked.bool(True)))

