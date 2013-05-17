import FWCore.ParameterSet.Config as cms
import sys


process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:dqm_file1.root","file:dqm_file3.root"))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_merged_file1_file3.root"))

process.e = cms.EndPath(process.out)


process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

if len(sys.argv) > 2:
    if sys.argv[2] == "Collate": 
        print "Collating option for multirunH"
        process.DQMStore.collateHistograms = cms.untracked.bool(True)


