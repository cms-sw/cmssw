import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST3")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testThinningTest2.root')
  #secondaryFileNames = cms.untracked.vstring('file:testSeriesOfProcessesPROD1.root')
)

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.thinningThingProducerE2 = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerD2alias'),
    trackTag = cms.InputTag('trackOfThingsProducerE2'),
    offsetToThinnedKey = cms.uint32(10),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerE2alias = cms.EDAlias(
  thinningThingProducerE2 = cms.VPSet(
    cms.PSet(type = cms.string('edmtestThings'))
  )
)

process.thinningThingProducerF2 = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerD2alias'),
    trackTag = cms.InputTag('trackOfThingsProducerF2'),
    offsetToThinnedKey = cms.uint32(10),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerF2alias = cms.EDAlias(
  thinningThingProducerF2 = cms.VPSet(
    cms.PSet(type = cms.string('edmtestThings'))
  )
)

process.testA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerA'),
    associationTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8),
    expectedValues = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8)
)

process.testB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerA'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8),
    expectedThinnedContent = cms.vint32(0, 1, 2, 3),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3),
    expectedValues = cms.vint32(0, 1, 2, 3)
)

process.testC = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerA'),
    thinnedTag = cms.InputTag('thinningThingProducerC'),
    associationTag = cms.InputTag('thinningThingProducerC'),
    trackTag = cms.InputTag('trackOfThingsProducerC'),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8),
    expectedThinnedContent = cms.vint32(4, 5, 6, 7),
    expectedIndexesIntoParent = cms.vuint32(4, 5, 6, 7),
    expectedValues = cms.vint32(4, 5, 6, 7)
)

process.testD = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerD', '', 'PROD'),
    associationTag = cms.InputTag('thinningThingProducerD', '', 'PROD'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    expectedIndexesIntoParent = cms.vuint32(10, 11, 12, 13, 14, 15, 16, 17, 18),
    expectedValues = cms.vint32(10, 11, 12, 13, 14, 15, 16, 17, -1)
)

process.testDPlus = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerD', '', 'PROD'),
    associationTag = cms.InputTag('thinningThingProducerD', '', 'PROD'),
    trackTag = cms.InputTag('trackOfThingsProducerDPlus'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    expectedIndexesIntoParent = cms.vuint32(10, 11, 12, 13, 14, 15, 16, 17, 18),
    expectedValues = cms.vint32(10, 11, 12, 13, 14, 15, 16, 17, -1, 21)
)

process.testE = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerD', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerE'),
    associationTag = cms.InputTag('thinningThingProducerE'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(10, 11, 12, 13, 14),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3, 4),
    expectedValues = cms.vint32(10, 11, 12, 13, 14)
)

process.testF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerD', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerF'),
    associationTag = cms.InputTag('thinningThingProducerF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(14, 15, 16, 17),
    expectedIndexesIntoParent = cms.vuint32(4, 5, 6, 7),
    expectedValues = cms.vint32(14, 15, 16, 17)
)

process.testG = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerG'),
    associationTag = cms.InputTag('thinningThingProducerG'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(20, 21, 22, 23, 24, 25, 26, 27, 28),
    expectedIndexesIntoParent = cms.vuint32(20, 21, 22, 23, 24, 25, 26, 27, 28),
    expectedValues = cms.vint32(20, 21, 22, 23, 24, 25, 26, 27, 28)
)

process.testH = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerG'),
    thinnedTag = cms.InputTag('thinningThingProducerH'),
    associationTag = cms.InputTag('thinningThingProducerH'),
    trackTag = cms.InputTag('trackOfThingsProducerH'),
    thinnedWasDropped = cms.bool(True),
    expectedParentContent = cms.vint32( 20,  21,  22,  23,  24,  25,  26,  27,  28),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(20, 21, 22, 23)
)

process.testI = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerG'),
    thinnedTag = cms.InputTag('thinningThingProducerI'),
    associationTag = cms.InputTag('thinningThingProducerI'),
    trackTag = cms.InputTag('trackOfThingsProducerI'),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedParentContent = cms.vint32( 20,  21,  22,  23,  24,  25,  26,  27,  28),
    expectedValues = cms.vint32(24, 25, 26, 27)
)

process.testJ = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerJ'),
    associationTag = cms.InputTag('thinningThingProducerJ'),
    trackTag = cms.InputTag('trackOfThingsProducerJ'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(-1, -1, -1, -1, -1, -1, -1, -1, -1)
)

process.testK = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerJ'),
    thinnedTag = cms.InputTag('thinningThingProducerK'),
    associationTag = cms.InputTag('thinningThingProducerK'),
    trackTag = cms.InputTag('trackOfThingsProducerK'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(-1, -1, -1, -1)
)

process.testL = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerJ'),
    thinnedTag = cms.InputTag('thinningThingProducerL'),
    associationTag = cms.InputTag('thinningThingProducerL'),
    trackTag = cms.InputTag('trackOfThingsProducerL'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(-1, -1, -1, -1)
)

process.testM = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerM'),
    associationTag = cms.InputTag('thinningThingProducerM'),
    trackTag = cms.InputTag('trackOfThingsProducerM'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    expectedIndexesIntoParent = cms.vuint32(40, 41, 42, 43, 44, 45, 46, 47, 48),
    expectedValues = cms.vint32(-1, -1, -1, -1, 44, 45, 46, 47, -1)
)

process.testN = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerM'),
    thinnedTag = cms.InputTag('thinningThingProducerN'),
    associationTag = cms.InputTag('thinningThingProducerN'),
    trackTag = cms.InputTag('trackOfThingsProducerN'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(-1, -1, -1, -1)
)

process.testO = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerM'),
    thinnedTag = cms.InputTag('aliasO'),
    associationTag = cms.InputTag('thinningThingProducerO'),
    trackTag = cms.InputTag('trackOfThingsProducerO'),
    parentWasDropped = cms.bool(True),
    thinnedIsAlias = cms.bool(True),
    expectedThinnedContent = cms.vint32(44, 45, 46, 47),
    expectedIndexesIntoParent = cms.vuint32(4, 5, 6, 7),
    expectedValues = cms.vint32(44, 45, 46, 47)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testThinningTest3.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer2alias_*_*',
        'drop *_thinningThingProducerD2alias_*_*',
        'drop *_thinningThingProducerE2_*_*',
        'drop *_thinningThingProducerF2_*_*',
        'drop *_thinningThingProducerA_*_*',
        'drop *_thinningThingProducerB_*_*',
        'drop *_thinningThingProducerC_*_*',
        'drop *_thinningThingProducerE_*_*',
        'drop *_thinningThingProducerF_*_*',
        'drop *_thinningThingProducerG_*_*',
        'drop *_thinningThingProducerO_*_*'
    )
)

process.p = cms.Path(process.thinningThingProducerE2 *
                     process.thinningThingProducerF2 *
                     process.testA *
                     process.testB *
                     process.testC *
                     process.testD *
                     process.testDPlus *
                     process.testE *
                     process.testF *
                     process.testG *
                     process.testH *
                     process.testI *
                     process.testJ *
                     process.testK *
                     process.testL *
                     process.testM *
                     process.testN *
                     process.testO
)

process.endPath = cms.EndPath(process.out)
