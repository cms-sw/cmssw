import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTMODULEDELETE")

process.maxEvents.input = 4
process.source = cms.Source("EmptySource")

# In presence of looper nothing is deleted
process.producerEventNotConsumed = cms.EDProducer("edmtest::MustRunIntProducer", ivalue = cms.int32(1), mustRunEvent = cms.bool(False))

process.pInt = cms.EDProducer("IntProducer",
    ivalue = cms.int32(1)
)
#Dummy looper will loop twice
process.looper = cms.Looper("TestModuleChangeLooper",
    startingValue = cms.untracked.int32(1),
    tag = cms.untracked.InputTag("pInt")
)

process.t1 = cms.Task(process.producerEventNotConsumed)
process.p1 = cms.Path(process.pInt, process.t1)
