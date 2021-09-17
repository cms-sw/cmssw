from builtins import range
import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            reScope = cms.untracked.string(""),
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
                                           runs  = cms.untracked.vint32([1, 2]),
                                           lumis = cms.untracked.vint32([0, 0]),
                                           means = cms.untracked.vdouble([i, i+1]),
                                           entries=cms.untracked.vdouble([1, 1])
 ))

readLumiElements=list()
for i in range(0,10):
 readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                           runs  = cms.untracked.vint32([1 for x in range(0,10)] + [2 for x in range(0,10)]),
                                           lumis = cms.untracked.vint32([x+1 for x in range(0,10)] + [x+1 for x in range(0,10)]),
                                           #file3 has means shifted by 1
                                           means = cms.untracked.vdouble([i for x in range(0,10)] + [i+1 for x in range(0,10)]),
                                           entries=cms.untracked.vdouble([1 for x in range(0,20)])
 ))

process.reader = cms.EDAnalyzer("DummyReadDQMStore",
                                runElements = cms.untracked.VPSet(*readRunElements),
                                lumiElements = cms.untracked.VPSet(*readLumiElements) )

process.e = cms.EndPath(process.check+process.reader)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

