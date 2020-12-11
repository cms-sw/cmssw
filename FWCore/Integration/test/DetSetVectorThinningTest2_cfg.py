import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testDetSetVectorThinningTest1.root')
)

process.slimmingTestA = cms.EDAnalyzer("ThinningDSVTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('slimmingThingProducerA'),
    associationTag = cms.InputTag('slimmingThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    parentWasDropped = cms.bool(True),
    thinnedSlimmedCount = cms.int32(1),
    refSlimmedCount = cms.int32(1),
    expectedParentContent = cms.VPSet(
        cms.PSet(id = cms.uint32(1), values = cms.vint32(range(0,50))),
        cms.PSet(id = cms.uint32(2), values = cms.vint32(range(50,100))),
        cms.PSet(id = cms.uint32(3), values = cms.vint32(range(100,150))),
    ),
    expectedThinnedContent = cms.VPSet(
        cms.PSet(id = cms.uint32(1), values = cms.vint32(range(0,9))),
        cms.PSet(id = cms.uint32(2), values = cms.vint32(range(50,59))),
        cms.PSet(id = cms.uint32(3), values = cms.vint32(range(100,109))),
    ),
    expectedIndexesIntoParent = cms.vuint32(
        list(range(0,9)) +
        list(range(50,59)) +
        list(range(100,109))
    ),
    expectedNumberOfTracks = cms.uint32(8*3),
    expectedValues = cms.vint32(
        list(range(0,9)) +
        list(range(50,59)) +
        list(range(100,109))
    )
)

process.p = cms.Path(
    process.slimmingTestA
)
