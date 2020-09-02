import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2A")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1A.root')
)

process.testA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerA'),
    associationTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    parentWasDropped = cms.bool(True),
    expectedParentContent = cms.vint32(range(0,50)),
    expectedThinnedContent = cms.vint32(range(0,20)),
    expectedIndexesIntoParent = cms.vuint32(range(0,20)),
    expectedValues = cms.vint32(range(0,20)),
)

process.testABF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerAB'),
    thinnedTag = cms.InputTag('thinningThingProducerABF'),
    associationTag = cms.InputTag('thinningThingProducerABF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(6,9)),
    expectedValues = cms.vint32(range(6,9)),
)

process.testB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    parentWasDropped = cms.bool(True),
    thinnedSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(0,50)),
    expectedThinnedContent = cms.vint32(range(0,11)),
    expectedIndexesIntoParent = cms.vuint32(range(0,11)),
    expectedValues = cms.vint32(range(0,11)),
)

process.testBD = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBD'),
    associationTag = cms.InputTag('thinningThingProducerBD'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(0,6)),
    expectedIndexesIntoParent = cms.vuint32(range(0,6)),
    expectedValues = cms.vint32(range(0,6)),
)

process.testBE = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBE'),
    associationTag = cms.InputTag('thinningThingProducerBE'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(2),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,11)),
    expectedIndexesIntoParent = cms.vuint32(range(6,11)),
    expectedValues = cms.vint32(range(6,11)),
)

process.testBEG = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerBE'),
    thinnedTag = cms.InputTag('thinningThingProducerBEG'),
    associationTag = cms.InputTag('thinningThingProducerBEG'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(9,11)),
    expectedIndexesIntoParent = cms.vuint32(range(3,5)),
    expectedValues = cms.vint32(range(9,11)),
)

process.p = cms.Path(
    process.testA
    * process.testABF
    * process.testB
    * process.testBD
    * process.testBE
    * process.testBEG
)
