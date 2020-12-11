import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2D")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1D.root')
)

process.testABE = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerAB'),
    thinnedTag = cms.InputTag('thinningThingProducerABE'),
    associationTag = cms.InputTag('thinningThingProducerABE'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    parentWasDropped = cms.bool(True),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,11)),
    expectedIndexesIntoParent = cms.vuint32(range(6,11)),
    expectedValues = cms.vint32(range(6,11)),
)

process.testBD = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBD'),
    associationTag = cms.InputTag('thinningThingProducerBD'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(1),
    refToParentIsAvailable = cms.bool(False),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(0,6)),
    expectedIndexesIntoParent = cms.vuint32(range(0,6)),
    expectedValues = cms.vint32(range(0,6)),
)

process.outD = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest2D.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerABE_*_*',
    )
)

process.p = cms.Path(
    process.testABE
    * process.testBD
)

process.ep = cms.EndPath(
    process.outD
)
