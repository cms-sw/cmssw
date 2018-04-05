import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(
    cms.InputTag("a"),
    cms.InputTag("b"),
    cms.InputTag("c"),
    cms.InputTag("d"),
    cms.InputTag("e"),
    cms.InputTag("f"),
    cms.InputTag("g"),
    cms.InputTag("h"),
    cms.InputTag("i"),
    cms.InputTag("j")
  ),
  expectedSum = cms.untracked.int32(30)
)

process.a = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.b = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.t1 = cms.Task(cms.TaskPlaceholder("c"))
process.t2 = cms.Task(process.a, cms.TaskPlaceholder("d"), process.t1)
process.t3 = cms.Task(cms.TaskPlaceholder("e"))
process.path1 = cms.Path(process.b, process.t2, process.t3)
process.t5 = cms.Task(process.a, cms.TaskPlaceholder("g"), cms.TaskPlaceholder("t4"))
process.t4 = cms.Task(cms.TaskPlaceholder("f"))
process.endpath1 = cms.EndPath(process.b + process.a1, process.t5)
process.t6 = cms.Task(cms.TaskPlaceholder("h"))
process.t7 = cms.Task(process.a, cms.TaskPlaceholder("i"), process.t6)
process.t8 = cms.Task(cms.TaskPlaceholder("j"))
process.schedule = cms.Schedule(process.path1, process.endpath1,tasks=[process.t7,process.t8])

process.c = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.d = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.e = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.f = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.g = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.h = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.i = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.j = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

