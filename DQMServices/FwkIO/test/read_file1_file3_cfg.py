from builtins import range
import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:dqm_file1.root","file:dqm_file3.root"))

seq = cms.untracked.VEventID()
for r in [1,2]:
    #begin run
    seq.append(cms.EventID(r,0,0))
    for l in range(1,11):
        #begin lumi
        seq.append(cms.EventID(r,l,0))
        #end lumi
        seq.append(cms.EventID(r,l,0))
    #end run
    seq.append(cms.EventID(r,0,0))

process.check = cms.EDAnalyzer("RunLumiEventChecker",
                               eventSequence = seq)

readRunElements = list()
for i in range(0,10):
 readRunElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                           means = cms.untracked.vdouble([i+x for x in (0,1)]),
                                           entries=cms.untracked.vdouble([1 for x in (0,1)])
 ))

readLumiElements=list()
for i in range(0,10):
 readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                           #file3 has means shifted by 1
                                           means = cms.untracked.vdouble([i+x//10 for x in range(0,20)]),
                                           entries=cms.untracked.vdouble([1 for x in range(0,20)])
 ))

process.reader = cms.EDAnalyzer("DummyReadDQMStore",
                                runElements = cms.untracked.VPSet(*readRunElements),
                                lumiElements = cms.untracked.VPSet(*readLumiElements) )

process.e = cms.EndPath(process.check+process.reader)

process.add_(cms.Service("DQMStore", forceResetOnBeginLumi = cms.untracked.bool(True)))
#process.add_(cms.Service("Tracer"))

