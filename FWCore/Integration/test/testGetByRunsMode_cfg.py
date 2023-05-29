import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:testGetBy1.root'
    ),
    processingMode=cms.untracked.string('Runs')
)

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(10000))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(100000))

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag(),
  expectedSum = cms.untracked.int32(110000),
  inputTagsBeginProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerBeginProcessBlock"),
  ),
  inputTagsEndProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock"),
  ),
  inputTagsEndProcessBlock2 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "two"),
  ),
  inputTagsEndProcessBlock3 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "three"),
  ),
  inputTagsEndProcessBlock4 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "four"),
  )
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGetByRunsMode.root')
)

process.p1 = cms.Path(process.intProducerBeginProcessBlock +
                      process.intProducerEndProcessBlock +
                      process.a1
)

process.e1 = cms.EndPath(process.out)
