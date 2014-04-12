import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:dqm_merged_file1_file2.root"))

seq = cms.untracked.VEventID()
lumisPerRun = [21,]
for r in [1,]:
    #begin run
    seq.append(cms.EventID(r,0,0))
    for l in xrange(1,lumisPerRun[r-1]):
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
                                          means = cms.untracked.vdouble(i),
                                          entries=cms.untracked.vdouble(2)
                                          ))

readLumiElements=list()
for i in xrange(0,10):
    readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                          means = cms.untracked.vdouble([i for x in xrange(0,20)]),
                                          entries=cms.untracked.vdouble([1 for x in xrange(0,20)])
                                          ))

process.reader = cms.EDAnalyzer("DummyReadDQMStore",
                               runElements = cms.untracked.VPSet(*readRunElements),
                               lumiElements = cms.untracked.VPSet(*readLumiElements) )

process.e = cms.EndPath(process.check+process.reader)

process.add_(cms.Service("DQMStore"))
#process.add_(cms.Service("Tracer"))

