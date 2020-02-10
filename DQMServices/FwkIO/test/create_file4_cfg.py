from builtins import range
import FWCore.ParameterSet.Config as cms
process =cms.Process("TEST")

process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(100),
                            firstLuminosityBlock = cms.untracked.uint32(100),
                            firstEvent = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))

elements = list()
for i in range(0,10):
    elements.append(cms.untracked.PSet(lowX=cms.untracked.double(0),
                                       highX=cms.untracked.double(11),
                                       nchX=cms.untracked.int32(11),
                                       name=cms.untracked.string("Foo"+str(i)),
                                       title=cms.untracked.string("Foo"+str(i)),
                                       value=cms.untracked.double(i)))

# A dummy tracked parameter is added to force the ProcessHistoryID to be different
# It serves no other purpose.
process.filler = cms.EDProducer("DummyFillDQMStore",
                                elements=cms.untracked.VPSet(*elements),
                                fillRuns = cms.untracked.bool(True),
                                fillLumis = cms.untracked.bool(True),
                                dummy = cms.bool(True))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_file4.root"))

readRunElements = list()
for i in range(0,10):
    readRunElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                              runs  = cms.untracked.vint32(1),
                                              lumis = cms.untracked.vint32(0),
                                              means = cms.untracked.vdouble(i),
                                              entries=cms.untracked.vdouble(1)
    ))

readLumiElements=list()
for i in range(0,10):
    readLumiElements.append(cms.untracked.PSet(name=cms.untracked.string("Foo"+str(i)),
                                              runs  = cms.untracked.vint32([1 for x in range(0,10)]),
                                              lumis = cms.untracked.vint32([x+100 for x in range(0,10)]),
                                              means = cms.untracked.vdouble([i for x in range(0,10)]),
                                              entries=cms.untracked.vdouble([1 for x in range(0,10)])
    ))

process.reader = cms.EDAnalyzer("DummyReadDQMStore",
                                runElements = cms.untracked.VPSet(*readRunElements),
                                lumiElements = cms.untracked.VPSet(*readLumiElements) )

process.p = cms.Path(process.filler)

process.o = cms.EndPath(process.out+process.reader)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.add_(cms.Service("DQMStore"))

