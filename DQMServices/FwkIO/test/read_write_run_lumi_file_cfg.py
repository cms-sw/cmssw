import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            reScope = cms.untracked.string(""),
                            fileNames = cms.untracked.vstring("file:dqm_run_lumi.root"))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_run_lumi_copy.root"))


process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

