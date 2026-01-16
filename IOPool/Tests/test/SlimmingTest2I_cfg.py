import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2I")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1I.root')
)

process.testB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    parentWasDropped = cms.bool(True),
    refToParentIsAvailable = cms.bool(False),
    thinnedSlimmedCount = cms.int32(1),
    expectedThinnedContent = cms.vint32(range(0,11)),
    expectedIndexesIntoParent = cms.vuint32(range(0,11)),
    expectedValues = cms.vint32(range(0,11)),
)

process.testCI = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerC'),
    thinnedTag = cms.InputTag('thinningThingProducerCI'),
    associationTag = cms.InputTag('thinningThingProducerCI'),
    trackTag = cms.InputTag('trackOfThingsProducerI'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(range(11,16)),
    expectedIndexesIntoParent = cms.vuint32(range(0,5)),
    expectedValues = cms.vint32(range(11,16)),
)

process.outI = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest2I.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerCI_*_*',
    )
)

process.p = cms.Path(
    process.testB
    * process.testCI
)

process.ep = cms.EndPath(
    process.outI
)
