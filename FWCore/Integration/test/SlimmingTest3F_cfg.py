import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3F")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest2F.root')
)

process.testABEF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerABE'),
    thinnedTag = cms.InputTag('thinningThingProducerABEF'),
    associationTag = cms.InputTag('thinningThingProducerABEF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    thinnedSlimmedCount = cms.int32(1),
    refSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(0,3)),
    expectedValues = cms.vint32(range(6,9)),
)

process.outF = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest3F.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
    )
)

process.p = cms.Path(
    process.testABEF
)

process.ep = cms.EndPath(
    process.outF
)
