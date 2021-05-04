import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2F")

process.maxEvents.input = 3

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1F.root')
)

process.thinningThingProducerABEF = cms.EDProducer("SlimmingThingProducer",
    inputTag = cms.InputTag('thinningThingProducerABE'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    offsetToThinnedKey = cms.uint32(6),
    offsetToValue = cms.uint32(6),
    expectedCollectionSize = cms.uint32(5)
)

process.testABEF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerABE'),
    thinnedTag = cms.InputTag('thinningThingProducerABEF'),
    associationTag = cms.InputTag('thinningThingProducerABEF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    thinnedSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(0,3)),
    expectedValues = cms.vint32(range(6,9)),
)

process.outF = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest2F.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_thinningThingProducerABEF_*_*',
        'keep *_trackOfThingsProducerF_*_*',
    )
)

process.p = cms.Path(
    process.thinningThingProducerABEF
    * process.testABEF
)

process.ep = cms.EndPath(
    process.outF
)
