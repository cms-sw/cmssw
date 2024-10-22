import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTMODULEDELETE")

process.maxEvents.input = 1
process.options.deleteNonConsumedUnscheduledModules = False
process.source = cms.Source("EmptySource")

process.intEventProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.intEventConsumer = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intEventProducer"),
    valueMustMatch = cms.untracked.int32(1)
)
process.intProducerNotConsumed = cms.EDProducer("edmtest::MustRunIntProducer",
    ivalue = cms.int32(1),
    mustRunEvent = cms.bool(False)
)

process.t = cms.Task(
    process.intEventProducer,
    process.intProducerNotConsumed
)
process.p = cms.Path(
    process.intEventConsumer,
    process.t
)
