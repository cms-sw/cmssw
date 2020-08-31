import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3B")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest2B.root')
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
    expectedParentContent = cms.vint32(range(0,10)),
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
    refToParentIsAvailable = cms.bool(False),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    refSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(9,11)),
    expectedIndexesIntoParent = cms.vuint32(range(3,5)),
    expectedValues = cms.vint32(range(9,11)),
)

process.outB = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest3B.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerBF_*_*',
    )
)

process.p = cms.Path(
    process.testBF
    * process.testBEG
)

process.ep = cms.EndPath(
    process.outB
)
