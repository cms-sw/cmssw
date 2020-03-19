import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            reScope = cms.untracked.string(""),
                            fileNames = cms.untracked.vstring("file:dqm_file1.root",
                                                              "file:dqm_file3.root",
                                                              "file:dqm_file2.root"),
                            filterOnRun = cms.untracked.uint32(1))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_merged_file1_file3_file2_filterOnRun1.root"),
                               filterOnRun = cms.untracked.uint32(1))
process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore", forceResetOnBeginLumi = cms.untracked.bool(True)))
#process.add_(cms.Service("Tracer"))

