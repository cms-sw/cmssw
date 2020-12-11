import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2H")

process.maxEvents.input = 3

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1H.root')
)

process.testH = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerH'),
    associationTag = cms.InputTag('thinningThingProducerH'),
    trackTag = cms.InputTag('trackOfThingsProducerH'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    refToParentIsAvailable = cms.bool(False),
    expectedValues = cms.vint32(range(0,19)),
)

process.p = cms.Path(
    process.testH
)
