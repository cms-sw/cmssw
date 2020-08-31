import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2B")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1B.root')
)

process.testABF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerAB'),
    thinnedTag = cms.InputTag('thinningThingProducerABF'),
    associationTag = cms.InputTag('thinningThingProducerABF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(6,9)),
    expectedValues = cms.vint32(range(6,9)),
)

process.testABG = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerAB'),
    thinnedTag = cms.InputTag('thinningThingProducerABG'),
    associationTag = cms.InputTag('thinningThingProducerABG'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    parentWasDropped = cms.bool(True),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(9,11)),
    expectedIndexesIntoParent = cms.vuint32(range(9,11)),
    expectedValues = cms.vint32(range(9,11)),
)

process.testBF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBF'),
    associationTag = cms.InputTag('thinningThingProducerBF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(6,9)),
    expectedValues = cms.vint32(range(6,9)),
)

process.testBEG = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerBE'),
    thinnedTag = cms.InputTag('thinningThingProducerBEG'),
    associationTag = cms.InputTag('thinningThingProducerBEG'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(9,11)),
    expectedIndexesIntoParent = cms.vuint32(range(3,5)),
    expectedValues = cms.vint32(range(9,11)),
)

process.outB = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest2B.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerABF_*_*',
        'drop *_thinningThingProducerABG_*_*',
    )
)

process.p = cms.Path(
    process.testABF
    * process.testABG
    * process.testBF
    * process.testBEG
)

process.ep = cms.EndPath(
    process.outB
)
