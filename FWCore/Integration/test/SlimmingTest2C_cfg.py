import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2C")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1C.root')
)

process.testBF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBF'),
    associationTag = cms.InputTag('thinningThingProducerBF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(1),
    refSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(6,9)),
    expectedValues = cms.vint32(range(6,9)),
)

process.testBEF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerBE'),
    thinnedTag = cms.InputTag('thinningThingProducerBEF'),
    associationTag = cms.InputTag('thinningThingProducerBEF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    refSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(0,3)),
    expectedValues = cms.vint32(range(6,9)),
)

process.outC = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest2C.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerBF_*_*',
    )
)

process.p = cms.Path(
    process.testBF
    * process.testBEF
)

process.ep = cms.EndPath(
    process.outC
)
