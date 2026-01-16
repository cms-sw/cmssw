import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2E")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1E.root')
)

process.testBD = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBD'),
    associationTag = cms.InputTag('thinningThingProducerBD'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(1),
    refSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(0,6)),
    expectedIndexesIntoParent = cms.vuint32(range(0,6)),
    expectedValues = cms.vint32(range(0,6)),
)

process.testBEF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerBE'),
    thinnedTag = cms.InputTag('thinningThingProducerBEF'),
    associationTag = cms.InputTag('thinningThingProducerBEF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    refToParentIsAvailable = cms.bool(False),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(0,3)),
    expectedValues = cms.vint32(range(6,9)),
)

process.outE = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest2E.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerBD_*_*',
    )
)

process.p = cms.Path(
    process.testBD
    * process.testBEF
)

process.ep = cms.EndPath(
    process.outE
)
