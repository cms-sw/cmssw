import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3E")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest2E.root')
)

process.testBEF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerBE'),
    thinnedTag = cms.InputTag('thinningThingProducerBEF'),
    associationTag = cms.InputTag('thinningThingProducerBEF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    refSlimmedCount = cms.int32(2),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(0,3)),
    expectedValues = cms.vint32(range(6,9)),
)

process.p = cms.Path(
    process.testBEF
)
