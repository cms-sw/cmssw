import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("DelayedReaderThrowingSource", labels = cms.untracked.vstring("fail"))

process.options.TryToContinue = ['TEST']
process.maxEvents.input = 3

process.intProd = cms.EDProducer("IntProducer", ivalue = cms.int32(10))
process.dependentAnalyzer = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("fail")),
    expectedSum = cms.untracked.int32(30)
)

process.dependent2 = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    inputTagsNotFound = cms.untracked.VInputTag( cms.InputTag("fail")),
    expectedSum = cms.untracked.int32(30)
)

process.independent = cms.EDAnalyzer("TestFindProduct",
    inputTags = cms.untracked.VInputTag(["intProd"]),
    expectedSum = cms.untracked.int32(30)
)

process.f = cms.EDFilter("IntProductFilter", label = cms.InputTag("intProd"))

process.options.modulesToCallForTryToContinue = [process.dependentAnalyzer.label_(), process.dependent2.label_()]

process.p = cms.Path(process.dependentAnalyzer, cms.Task(process.intProd))
process.p2 = cms.Path(cms.wait(process.dependent2)+process.f+process.independent)
#process.add_(cms.Service("Tracer"))
