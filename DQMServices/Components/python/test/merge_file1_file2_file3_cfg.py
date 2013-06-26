import FWCore.ParameterSet.Config as cms
import sys


process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:dqm_file1.root","file:dqm_file2.root","file:dqm_file3.root"))

if len(sys.argv) > 2:
    myFileName = "dqm_merged_file1_file2_file3_run%s.root" % (sys.argv[2])
else:
    myFileName = "dqm_merged_file1_file2_file3.root"
process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string(myFileName))

process.e = cms.EndPath(process.out)


process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

if len(sys.argv) > 2:
    if sys.argv[2] != 1: 
        print "selecting on run %s"% (sys.argv[2])
        process.source.filterOnRun = cms.untracked.uint32(int(sys.argv[2]))
        process.out.filterOnRun = cms.untracked.uint32(int(sys.argv[2]))

