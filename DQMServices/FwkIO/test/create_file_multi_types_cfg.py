from builtins import range
import FWCore.ParameterSet.Config as cms
process =cms.Process("TEST")

process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(100),
                            firstLuminosityBlock = cms.untracked.uint32(1),
                            firstEvent = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1))

elements = list()
extensions = ["","2D"]
for t in [0,1]:
    for i in range(0,10):
        elements.append(cms.untracked.PSet(type = cms.untracked.uint32(t+1),
                                           lowX=cms.untracked.double(0),
                                           highX=cms.untracked.double(10),
                                           nchX=cms.untracked.int32(10),
                                           lowY=cms.untracked.double(0),
                                           highY=cms.untracked.double(10),
                                           nchY=cms.untracked.int32(2),
                                           name=cms.untracked.string("Foo"+extensions[t]+str(i)),
                                           title=cms.untracked.string("Foo"+str(i)),
                                           value=cms.untracked.double(i)))

process.filler = cms.EDProducer("DummyFillDQMStore",
                                elements=cms.untracked.VPSet(*elements),
                                fillRuns = cms.untracked.bool(True),
                                fillLumis = cms.untracked.bool(True))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_file_multi_types.root"))

process.p = cms.Path(process.filler)

process.o = cms.EndPath(process.out)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.add_(cms.Service("DQMStore"))

