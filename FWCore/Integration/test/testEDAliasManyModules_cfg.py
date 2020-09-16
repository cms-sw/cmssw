import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents.input = 1
process.source = cms.Source("EmptySource")

process.intProducerFoo = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.intProducerBar = cms.EDProducer("IntProducer", ivalue = cms.int32(2))
process.intConsumer = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustMatch = cms.untracked.int32(1)
)
process.intConsumer2 = process.intConsumer.clone(
    moduleLabel = ("intProducer", "test"),
    valueMustMatch = 2
)

process.intProducer = cms.EDAlias(
    intProducerFoo = cms.VPSet(
        cms.PSet(
            type = cms.string("edmtestIntProduct"),
        )
    ),
    intProducerBar = cms.VPSet(
        cms.PSet(
            type = cms.string("edmtestIntProduct"),
            fromProductInstance = cms.string(""),
            toProductInstance = cms.string("test"),
        )
    )
)

process.PathsAndConsumesOfModulesTestService = cms.Service("PathsAndConsumesOfModulesTestService",
    modulesAndConsumes = cms.VPSet(
        cms.PSet(
            key = cms.string("intConsumer"),
            value = cms.vstring("intProducerFoo")
        ),
        cms.PSet(
            key = cms.string("intConsumer2"),
            value = cms.vstring("intProducerBar")
        )
    )
)

process.t = cms.Task(
    process.intProducerFoo,
    process.intProducerBar
)
process.p = cms.Path(
    process.intConsumer +
    process.intConsumer2,
    process.t
)
