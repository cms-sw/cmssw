import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.options.TryToContinue = ['NotFound']
process.maxEvents.input = 3

process.fail = cms.EDProducer("FailingProducer")
process.intProd = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
process.dependentAnalyzer = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("fail")),
    expectedSum = cms.untracked.int32(0)
)

process.p = cms.Path(process.dependentAnalyzer, cms.Task(process.fail,process.intProd))

process.out = cms.OutputModule("SewerModule",
    name=cms.string("out"),
    shouldPass = cms.int32(3),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("exception@p")),
    outputCommands = cms.untracked.vstring("drop *", "keep *_intProd_*_*")
)

process.out.shouldTryToContinue()

#process.add_(cms.Service("Tracer"))
