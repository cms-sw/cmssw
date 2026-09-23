import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 3

process.externalWorkAllocProducer = cms.EDProducer("allocMonTest::ExternalWorkAllocProducer")
process.transformAllocProducer = cms.EDProducer("allocMonTest::TransformAllocProducer")
process.transformAsyncAllocProducer = cms.EDProducer("allocMonTest::TransformAsyncAllocProducer")
process.allOfTheAboveAllocProducer = cms.EDProducer("allocMonTest::ExternalWorkTransformAllocProducer")

process.out = cms.OutputModule("AsciiOutputModule")
process.ep = cms.EndPath(process.out, cms.Task(
    process.externalWorkAllocProducer,
    process.transformAllocProducer,
    process.transformAsyncAllocProducer,
    process.allOfTheAboveAllocProducer,
))

process.add_(cms.Service("ModuleAllocMonitor", fileName = cms.untracked.string("moduleAllocAcquireTransform.log")))
