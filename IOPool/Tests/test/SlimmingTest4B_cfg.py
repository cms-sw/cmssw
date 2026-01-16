import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST4B")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest3B.root')
)

process.testBEG = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerBE'),
    thinnedTag = cms.InputTag('thinningThingProducerBEG'),
    associationTag = cms.InputTag('thinningThingProducerBEG'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    parentWasDropped = cms.bool(True),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    refSlimmedCount = cms.int32(2),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(9,11)),
    expectedIndexesIntoParent = cms.vuint32(range(3,5)),
    expectedValues = cms.vint32(range(9,11)),
)

process.p = cms.Path(
    process.testBEG
)
