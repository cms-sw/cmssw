import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3D")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest2D.root')
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

process.p = cms.Path(
    process.testBD
)
