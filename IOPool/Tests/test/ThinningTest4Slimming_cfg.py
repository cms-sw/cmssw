import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST4")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testThinningTest3Slimming.root')
)

process.slimmingTestA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer', '', 'PROD'),
    thinnedTag = cms.InputTag('slimmingThingProducerA'),
    associationTag = cms.InputTag('slimmingThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    parentWasDropped = cms.bool(True),
    thinnedSlimmedCount = cms.int32(1),
    refSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                       10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                       20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                       30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                       40, 41, 42, 43, 44, 45, 46, 47, 48, 49
    ),
    expectedThinnedContent = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8),
    expectedValues = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8)
)

process.p = cms.Path(process.slimmingTestA)
