import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.fail = cms.EDProducer("FailingProducer")

process.readFail = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag("fail"))

process.a = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(5), iterations = cms.uint32(10000))

process.p2 = cms.Path(process.fail)
process.p1 = cms.Path(process.readFail+process.a)

process.add_(cms.Service("Tracer"))
