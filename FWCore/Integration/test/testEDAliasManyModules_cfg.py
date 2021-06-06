import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents.input = 1
process.source = cms.Source("EmptySource")

###########
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

###########
process.intVecProducerFoo = cms.EDProducer("IntVectorProducer",
    count = cms.int32(10),
    ivalue = cms.int32(0),
    delta = cms.int32(1)
)
process.intVecProducerBar = process.intVecProducerFoo.clone(ivalue = 100)
process.intViewConsumer = cms.EDProducer("IntVecPtrVectorProducer",
    target = cms.InputTag('intVecProducer')
)
process.intViewConsumer2 = cms.EDProducer("IntVecPtrVectorProducer",
    target = cms.InputTag("intVecProducer", "test")
)

process.intVecProducer = cms.EDAlias(
    intVecProducerFoo = cms.VPSet(
        cms.PSet(
            type = cms.string("ints")
        )
    ),
    intVecProducerBar = cms.VPSet(
        cms.PSet(
            type = cms.string("ints"),
            fromProductInstance = cms.string(""),
            toProductInstance = cms.string("test"),
        )
    )
)

###########
process.PathsAndConsumesOfModulesTestService = cms.Service("PathsAndConsumesOfModulesTestService",
    modulesAndConsumes = cms.VPSet(
        cms.PSet(
            key = cms.string("intConsumer"),
            value = cms.vstring("intProducerFoo")
        ),
        cms.PSet(
            key = cms.string("intConsumer2"),
            value = cms.vstring("intProducerBar")
        ),
        cms.PSet(
            key = cms.string("intViewConsumer"),
            value = cms.vstring("intVecProducerFoo")
        ),
        cms.PSet(
            key = cms.string("intViewConsumer2"),
            value = cms.vstring("intVecProducerBar")
        ),
    )
)

process.t = cms.Task(
    process.intProducerFoo,
    process.intProducerBar,
    process.intVecProducerFoo,
    process.intVecProducerBar,
)
process.p = cms.Path(
    process.intConsumer +
    process.intConsumer2 +
    process.intViewConsumer +
    process.intViewConsumer2,
    process.t
)
