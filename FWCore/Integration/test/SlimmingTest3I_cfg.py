import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3I")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest2I.root')
)

process.testB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    parentWasDropped = cms.bool(True),
    thinnedSlimmedCount = cms.int32(1),
    refSlimmedCount = cms.int32(1),
    expectedThinnedContent = cms.vint32(range(0,11)),
    expectedIndexesIntoParent = cms.vuint32(range(0,11)),
    expectedValues = cms.vint32(range(0,11)),
)

process.p = cms.Path(
    process.testB
)
