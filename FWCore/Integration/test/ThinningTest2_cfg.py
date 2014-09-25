import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testThinningTest1.root')
)

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

# Produce some products that have the same branch name except process
# These test disambiguation when there are multiple entries
# in the ThinnedAssociationsHelper with the same parent BranchID.
process.thingProducer = cms.EDProducer("ThingProducer",
                                       offsetDelta = cms.int32(2000),
                                       nThings = cms.int32(50)
)

process.thinningThingProducerD = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.thinningThingProducerETEST = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerD'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    offsetToThinnedKey = cms.uint32(10),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerFTEST = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerD'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    offsetToThinnedKey = cms.uint32(10),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerD2 = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer2alias'),
    trackTag = cms.InputTag('trackOfThingsProducerD2'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.thinningThingProducerD2alias = cms.EDAlias(
  thinningThingProducerD2 = cms.VPSet(
    cms.PSet(type = cms.string('edmtestThings'))
  )
)

process.testA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer', '', 'PROD'),
    thinnedTag = cms.InputTag('thinningThingProducerA'),
    associationTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
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

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testThinningTest2.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer_*_*',
        'drop *_thinningThingProducerD2_*_*',
        'drop *_thinningThingProducerD_*_*',
        'drop *_thinningThingProducerH_*_*',
        'drop *_thinningThingProducerI_*_*',
        'drop *_thinningThingProducerJ_*_*',
        'drop *_thinningThingProducerK_*_*',
        'drop *_thinningThingProducerL_*_*',
        'drop *_aliasM_*_*',
        'drop *_aliasN_*_*',
    )
)

process.p = cms.Path(process.thinningThingProducerD2 * process.testA * process.testB * process.testC
                     * process.thingProducer
                     * process.thinningThingProducerD
                     * process.thinningThingProducerETEST
                     * process.thinningThingProducerFTEST)

process.endPath = cms.EndPath(process.out)
