import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTMODULEDELETE")

process.maxEvents.input = 8
process.source = cms.Source("EmptySource")

process.producerEventNotConsumed1 = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcEvent = cms.untracked.VInputTag("producerEventNotConsumed2")
)
process.producerEventNotConsumed2 = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcEvent = cms.untracked.VInputTag("producerEventNotConsumed3")
)
process.producerEventNotConsumed3 = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcEvent = cms.untracked.VInputTag("producerEventNotConsumed1")
)
process.consumerEvent = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer",
    srcEvent = cms.untracked.VInputTag("producerEventNotConsumed3")
)

process.t = cms.Task(
    process.producerEventNotConsumed1,
    process.producerEventNotConsumed2,
    process.producerEventNotConsumed3,
)
process.p = cms.Path(
    process.consumerEvent,
    process.t
)
