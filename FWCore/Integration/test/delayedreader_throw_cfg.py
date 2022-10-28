import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("DelayedReaderThrowingSource", labels = cms.untracked.vstring("test", "test2", "test3"))

process.getter = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("test","","INPUTTEST")))
process.onPath = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("test2", "", "INPUTTEST"), cms.InputTag("getter", "other")))
process.f1 = cms.EDFilter("IntProductFilter", label = cms.InputTag("onPath"), shouldProduce = cms.bool(True))
process.f2 = cms.EDFilter("IntProductFilter", label = cms.InputTag("onPath"), shouldProduce = cms.bool(True))
process.inFront = cms.EDFilter("IntProductFilter", label = cms.InputTag("test3"))

process.p1 = cms.Path(process.inFront+process.onPath+process.f1+process.f2)
process.p3 = cms.Path(process.onPath+process.f1, cms.Task(process.getter))

process.p2 = cms.Path(process.onPath+process.f2)


#process.dump = cms.EDAnalyzer("EventContentAnalyzer")
#process.p = cms.Path(process.dump)

process.out = cms.OutputModule("AsciiOutputModule")
process.e = cms.EndPath(process.out, cms.Task(process.getter))

process.maxEvents.input = 1

#process.add_(cms.Service("Tracer"))
