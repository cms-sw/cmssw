from builtins import range
import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            reScope = cms.untracked.string(""),
                            fileNames = cms.untracked.vstring("file:dqm_run_only.root"))

#NOTE: even though we only store histograms on runs, we still record the lumis that were seen
seq = cms.untracked.VEventID()
for r in range(1,11):
    #begin run
    seq.append(cms.EventID(r,0,0))
    for l in range(1,2):
        #begin lumi
        seq.append(cms.EventID(r,l,0))
        #end lumi
        seq.append(cms.EventID(r,l,0))
    #end run
    seq.append(cms.EventID(r,0,0))


process.check = cms.EDAnalyzer("RunLumiEventChecker",
                               eventSequence = seq)

process.e = cms.EndPath(process.check)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

