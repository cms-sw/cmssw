from builtins import range
import FWCore.ParameterSet.Config as cms
process =cms.Process("TEST")

process.source = cms.Source("EmptySource", numberEventsInRun = cms.untracked.uint32(1))

elements = list()
for i in range(0,10):
    elements.append(cms.untracked.PSet(lowX=cms.untracked.double(0),
                                       highX=cms.untracked.double(10),
                                       nchX=cms.untracked.int32(10),
                                       name=cms.untracked.string("Foo"+str(i)),
                                       title=cms.untracked.string("Foo"+str(i)),
                                       value=cms.untracked.double(i)))

process.filler = cms.EDProducer("DummyFillDQMStore",
                                elements=cms.untracked.VPSet(*elements),
                                fillRuns = cms.untracked.bool(False),
                                fillLumis = cms.untracked.bool(True))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("dqm_lumi_only.root"))

process.p = cms.Path(process.filler)

process.o = cms.EndPath(process.out)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.add_(cms.Service("DQMStore"))

