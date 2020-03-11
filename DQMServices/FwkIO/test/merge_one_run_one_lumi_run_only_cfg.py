import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            reScope = cms.untracked.string(""),
                            fileNames = cms.untracked.vstring("file:dqm_one_run_one_lumi_run_only.root","file:dqm_one_run_one_lumi_run_only_2.root"))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_merged_one_run_one_lumi_run_only.root"))
process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

