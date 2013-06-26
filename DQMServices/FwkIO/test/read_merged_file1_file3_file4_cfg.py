import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:dqm_merged_file1_file3_file4.root"))

seq = cms.untracked.VEventID()
lumisPerRun = [21,11]
r = 1
#begin run
seq.append(cms.EventID(r,0,0))
for l in xrange(1,11):
    #begin lumi
    seq.append(cms.EventID(r,l,0))
    #end lumi
    seq.append(cms.EventID(r,l,0))
#end run
seq.append(cms.EventID(r,0,0))
r = 2
#begin run
seq.append(cms.EventID(r,0,0))
for l in xrange(1,11):
    #begin lumi
    seq.append(cms.EventID(r,l,0))
    #end lumi
    seq.append(cms.EventID(r,l,0))
#end run
seq.append(cms.EventID(r,0,0))
r = 1
#begin run
seq.append(cms.EventID(r,0,0))
for l in xrange(100,110):
    #begin lumi
    seq.append(cms.EventID(r,l,0))
    #end lumi
    seq.append(cms.EventID(r,l,0))
#end run
seq.append(cms.EventID(r,0,0))

process.check = cms.EDAnalyzer("MulticoreRunLumiEventChecker",
                               eventSequence = seq)

readRunElements = list()
for i in xrange(0,10):
    readRunElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                          means = cms.untracked.vdouble([i+x for x in (0,1,0)]),
                                          entries=cms.untracked.vdouble([x for x in (1,1,1)])
                                          ))

readLumiElements=list()
for i in xrange(0,10):
    readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                          #file3, which is run 2 has means shifted by 1
                                          means = cms.untracked.vdouble([(i+x/10-x/20-x/20) for x in xrange(0,30)]),
                                          entries=cms.untracked.vdouble([1 for x in xrange(0,30)])
                                          ))

process.reader = cms.EDAnalyzer("DummyReadDQMStore",
                               runElements = cms.untracked.VPSet(*readRunElements),
                               lumiElements = cms.untracked.VPSet(*readLumiElements) )

process.e = cms.EndPath(process.check+process.reader)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

