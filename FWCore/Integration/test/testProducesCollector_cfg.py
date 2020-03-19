import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.producerUsingCollector = cms.EDProducer("ProducerUsingCollector")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testProducesCollector.root')
)

process.test = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(cms.InputTag("producerUsingCollector"),
                                      cms.InputTag("producerUsingCollector", "event"),
                                      cms.InputTag("producerUsingCollector", "eventOther")),
  inputTagsUInt64 = cms.untracked.VInputTag(cms.InputTag("producerUsingCollector")),
  inputTagsEndLumi = cms.untracked.VInputTag(cms.InputTag("producerUsingCollector", "beginLumi"),
                                             cms.InputTag("producerUsingCollector", "endLumi")),
  inputTagsEndRun = cms.untracked.VInputTag(cms.InputTag("producerUsingCollector", "beginRun"),
                                            cms.InputTag("producerUsingCollector", "endRun")),
  expectedSum = cms.untracked.int32(56)
)

process.p = cms.Path(process.producerUsingCollector * process.test)

process.e = cms.EndPath(process.out)
