# Test for thinning collections and redirecting Refs
# with SubProcesses. The basic strategy here is to
# repeat tests ThinningTest1_cfg.py, ThinningTest2_cfg.py
# and ThinningTest3_cfg.py except run them in one
# process with SubProcesses.  There is one possibly
# surprising feature here.  Refs to dropped products
# will still succeed to find elements in containers
# the were dropped in earlier SubProcesses even though
# a getByToken call will not find the product. This
# is because Refs contain direct pointers to the
# product or the ProductGetter and do not use
# the lookup mechanism that respects the product
# drops. So the expected values in the test
# below must all reflect that. We have to
# run a separate process to actually test that
# the redirection of Refs is working.
# See ThinningTestSubProcessRead_cfg.py.

import FWCore.ParameterSet.Config as cms

process = cms.Process("FIRST")

#process.Tracer = cms.Service('Tracer')

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO.limit = 100

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.thingProducer = cms.EDProducer("ThingProducer",
                                       offsetDelta = cms.int32(100),
                                       nThings = cms.int32(50)
)

process.trackOfThingsProducerA = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8)
)

process.trackOfThingsProducerB = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(0, 1, 2, 3)
)

process.trackOfThingsProducerC = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(4, 5, 6, 7)
)

process.trackOfThingsProducerD = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(10, 11, 12, 13, 14, 15, 16, 17, 18)
)

process.trackOfThingsProducerE = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(10, 11, 12, 13)
)

process.trackOfThingsProducerF = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(14, 15, 16, 17)
)

process.trackOfThingsProducerG = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(20, 21, 22, 23, 24, 25, 26, 27, 28)
)

process.trackOfThingsProducerH = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(20, 21, 22, 23)
)

process.trackOfThingsProducerI = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(24, 25, 26, 27)
)

process.trackOfThingsProducerJ = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(30, 31, 32, 33, 34, 35, 36, 37, 38)
)

process.trackOfThingsProducerK = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(30, 31, 32, 33)
)

process.trackOfThingsProducerL = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(34, 35, 36, 37)
)

process.trackOfThingsProducerM = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(40, 41, 42, 43, 44, 45, 46, 47, 48)
)

process.trackOfThingsProducerN = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(40, 41, 42, 43)
)

process.trackOfThingsProducerO = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(44, 45, 46, 47)
)

process.thinningThingProducerA = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.thinningThingProducerB = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerC = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerC'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerD = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.thinningThingProducerE = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerD'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    offsetToThinnedKey = cms.uint32(10),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerF = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerD'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    offsetToThinnedKey = cms.uint32(10),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerG = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.thinningThingProducerH = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerG'),
    trackTag = cms.InputTag('trackOfThingsProducerH'),
    offsetToThinnedKey = cms.uint32(20),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerI = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerG'),
    trackTag = cms.InputTag('trackOfThingsProducerI'),
    offsetToThinnedKey = cms.uint32(20),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerJ = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerJ'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.thinningThingProducerK = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerJ'),
    trackTag = cms.InputTag('trackOfThingsProducerK'),
    offsetToThinnedKey = cms.uint32(30),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerL = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerJ'),
    trackTag = cms.InputTag('trackOfThingsProducerL'),
    offsetToThinnedKey = cms.uint32(30),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerM = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerM'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.aliasM = cms.EDAlias(
  thinningThingProducerM = cms.VPSet(
    cms.PSet(type = cms.string('edmtestThings')),
    # the next one should get ignored
    cms.PSet(type = cms.string('edmThinnedAssociation')) 
  )
)

process.thinningThingProducerN = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerM'),
    trackTag = cms.InputTag('trackOfThingsProducerN'),
    offsetToThinnedKey = cms.uint32(40),
    expectedCollectionSize = cms.uint32(9)
)

process.aliasN = cms.EDAlias(
  thinningThingProducerN = cms.VPSet(
    cms.PSet(type = cms.string('edmtestThings')),
    # the next one should get ignored
    cms.PSet(type = cms.string('edmThinnedAssociation')) 
  )
)

process.thinningThingProducerO = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerM'),
    trackTag = cms.InputTag('trackOfThingsProducerO'),
    offsetToThinnedKey = cms.uint32(40),
    expectedCollectionSize = cms.uint32(9)
)

process.aliasO = cms.EDAlias(
  thinningThingProducerO = cms.VPSet(
    cms.PSet(type = cms.string('edmtestThings')),
    # the next one should get ignored
    cms.PSet(type = cms.string('edmThinnedAssociation')) 
  )
)

process.testFirstA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
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

process.testFirstB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerA'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8),
    expectedThinnedContent = cms.vint32(0, 1, 2, 3),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3),
    expectedValues = cms.vint32(0, 1, 2, 3)
)

process.testFirstC = cms.EDAnalyzer("ThinningTestAnalyzer",
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
    fileName = cms.untracked.string('testThinningTestSubProcess1.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerM_*_*',
        'drop *_thinningThingProducerN_*_*',
        'drop *_thinningThingProducerO_*_*'
    )
)

process.p = cms.Path(process.thingProducer * process.trackOfThingsProducerA
                                           * process.trackOfThingsProducerB
                                           * process.trackOfThingsProducerC
                                           * process.trackOfThingsProducerD
                                           * process.trackOfThingsProducerE
                                           * process.trackOfThingsProducerF
                                           * process.trackOfThingsProducerG
                                           * process.trackOfThingsProducerH
                                           * process.trackOfThingsProducerI
                                           * process.trackOfThingsProducerJ
                                           * process.trackOfThingsProducerK
                                           * process.trackOfThingsProducerL
                                           * process.trackOfThingsProducerM
                                           * process.trackOfThingsProducerN
                                           * process.trackOfThingsProducerO
                                           * process.thinningThingProducerA
                                           * process.thinningThingProducerB
                                           * process.thinningThingProducerC
                                           * process.thinningThingProducerD
                                           * process.thinningThingProducerE
                                           * process.thinningThingProducerF
                                           * process.thinningThingProducerG
                                           * process.thinningThingProducerH
                                           * process.thinningThingProducerI
                                           * process.thinningThingProducerJ
                                           * process.thinningThingProducerK
                                           * process.thinningThingProducerL
                                           * process.thinningThingProducerM
                                           * process.thinningThingProducerN
                                           * process.thinningThingProducerO
                                           * process.testFirstA
                                           * process.testFirstB
                                           * process.testFirstC
                    )

process.endPath = cms.EndPath(process.out)

# ---------------------------------------------------------------

secondProcess = cms.Process("SECOND")
process.subProcess = cms.SubProcess(secondProcess,
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thinningThingProducerM_*_*',
        'drop *_thinningThingProducerN_*_*',
        'drop *_thinningThingProducerO_*_*',
    )
)

secondProcess.testSecondA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
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

secondProcess.testSecondB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerA'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8),
    expectedThinnedContent = cms.vint32(0, 1, 2, 3),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3),
    expectedValues = cms.vint32(0, 1, 2, 3)
)

secondProcess.testSecondC = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerA'),
    thinnedTag = cms.InputTag('thinningThingProducerC'),
    associationTag = cms.InputTag('thinningThingProducerC'),
    trackTag = cms.InputTag('trackOfThingsProducerC'),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8),
    expectedThinnedContent = cms.vint32(4, 5, 6, 7),
    expectedIndexesIntoParent = cms.vuint32(4, 5, 6, 7),
    expectedValues = cms.vint32(4, 5, 6, 7)
)

secondProcess.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testThinningTestSubProcess2.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer_*_*',
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

secondProcess.p = cms.Path(secondProcess.testSecondA * secondProcess.testSecondB * secondProcess.testSecondC)

secondProcess.endPath = cms.EndPath(secondProcess.out)

# ---------------------------------------------------------------

thirdProcess = cms.Process("THIRD")
secondProcess.subProcess = cms.SubProcess(thirdProcess,
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer_*_*',
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

thirdProcess.testThirdA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerA'),
    associationTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8),
    expectedValues = cms.vint32(0, 1, 2, 3, 4, 5, 6, 7, 8)
)

thirdProcess.testThirdB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerA'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8),
    expectedThinnedContent = cms.vint32(0, 1, 2, 3),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3),
    expectedValues = cms.vint32(0, 1, 2, 3)
)

thirdProcess.testThirdC = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerA'),
    thinnedTag = cms.InputTag('thinningThingProducerC'),
    associationTag = cms.InputTag('thinningThingProducerC'),
    trackTag = cms.InputTag('trackOfThingsProducerC'),
    expectedParentContent = cms.vint32( 0,  1,  2,  3,  4,  5,  6,  7,  8),
    expectedThinnedContent = cms.vint32(4, 5, 6, 7),
    expectedIndexesIntoParent = cms.vuint32(4, 5, 6, 7),
    expectedValues = cms.vint32(4, 5, 6, 7)
)

thirdProcess.testThirdD = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerD'),
    associationTag = cms.InputTag('thinningThingProducerD'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    expectedIndexesIntoParent = cms.vuint32(10, 11, 12, 13, 14, 15, 16, 17, 18),
    expectedValues = cms.vint32(10, 11, 12, 13, 14, 15, 16, 17, 18)
)

thirdProcess.testThirdE = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerD'),
    thinnedTag = cms.InputTag('thinningThingProducerE'),
    associationTag = cms.InputTag('thinningThingProducerE'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(10, 11, 12, 13),
    expectedIndexesIntoParent = cms.vuint32(0, 1, 2, 3),
    expectedValues = cms.vint32(10, 11, 12, 13)
)

thirdProcess.testThirdF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerD'),
    thinnedTag = cms.InputTag('thinningThingProducerF'),
    associationTag = cms.InputTag('thinningThingProducerF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(14, 15, 16, 17),
    expectedIndexesIntoParent = cms.vuint32(4, 5, 6, 7),
    expectedValues = cms.vint32(14, 15, 16, 17)
)

thirdProcess.testThirdG = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerG'),
    associationTag = cms.InputTag('thinningThingProducerG'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(20, 21, 22, 23, 24, 25, 26, 27, 28),
    expectedIndexesIntoParent = cms.vuint32(20, 21, 22, 23, 24, 25, 26, 27, 28),
    expectedValues = cms.vint32(20, 21, 22, 23, 24, 25, 26, 27, 28)
)

thirdProcess.testThirdH = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerG'),
    thinnedTag = cms.InputTag('thinningThingProducerH'),
    associationTag = cms.InputTag('thinningThingProducerH'),
    trackTag = cms.InputTag('trackOfThingsProducerH'),
    thinnedWasDropped = cms.bool(True),
    expectedParentContent = cms.vint32( 20,  21,  22,  23,  24,  25,  26,  27,  28),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(20, 21, 22, 23)
)

thirdProcess.testThirdI = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerG'),
    thinnedTag = cms.InputTag('thinningThingProducerI'),
    associationTag = cms.InputTag('thinningThingProducerI'),
    trackTag = cms.InputTag('trackOfThingsProducerI'),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedParentContent = cms.vint32( 20,  21,  22,  23,  24,  25,  26,  27,  28),
    expectedValues = cms.vint32(24, 25, 26, 27)
)

thirdProcess.testThirdJ = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerJ'),
    associationTag = cms.InputTag('thinningThingProducerJ'),
    trackTag = cms.InputTag('trackOfThingsProducerJ'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(30, 31, 32, 33, 34, 35, 36, 37, 38)
)

thirdProcess.testThirdK = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerJ'),
    thinnedTag = cms.InputTag('thinningThingProducerK'),
    associationTag = cms.InputTag('thinningThingProducerK'),
    trackTag = cms.InputTag('trackOfThingsProducerK'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(30, 31, 32, 33)
)

thirdProcess.testThirdL = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerJ'),
    thinnedTag = cms.InputTag('thinningThingProducerL'),
    associationTag = cms.InputTag('thinningThingProducerL'),
    trackTag = cms.InputTag('trackOfThingsProducerL'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(34, 35, 36, 37)
)

thirdProcess.testThirdM = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerM'),
    associationTag = cms.InputTag('thinningThingProducerM'),
    trackTag = cms.InputTag('trackOfThingsProducerM'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    expectedIndexesIntoParent = cms.vuint32(40, 41, 42, 43, 44, 45, 46, 47, 48),
    expectedValues = cms.vint32(40, 41, 42, 43, 44, 45, 46, 47, 48)
)

thirdProcess.testThirdN = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerM'),
    thinnedTag = cms.InputTag('thinningThingProducerN'),
    associationTag = cms.InputTag('thinningThingProducerN'),
    trackTag = cms.InputTag('trackOfThingsProducerN'),
    parentWasDropped = cms.bool(True),
    thinnedWasDropped = cms.bool(True),
    associationShouldBeDropped = cms.bool(True),
    expectedValues = cms.vint32(40, 41, 42, 43)
)

thirdProcess.testThirdO = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerM'),
    thinnedTag = cms.InputTag('aliasO'),
    associationTag = cms.InputTag('thinningThingProducerO'),
    trackTag = cms.InputTag('trackOfThingsProducerO'),
    thinnedIsAlias = cms.bool(False), # See unaliased in SubProcess
    parentWasDropped = cms.bool(True),
    expectedThinnedContent = cms.vint32(44, 45, 46, 47),
    expectedIndexesIntoParent = cms.vuint32(4, 5, 6, 7),
    expectedValues = cms.vint32(44, 45, 46, 47)
)

thirdProcess.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testThinningTestSubProcess3.root')
)

thirdProcess.p = cms.Path(thirdProcess.testThirdA *
                     thirdProcess.testThirdB *
                     thirdProcess.testThirdC *
                     thirdProcess.testThirdD *
                     thirdProcess.testThirdE *
                     thirdProcess.testThirdF *
                     thirdProcess.testThirdG *
                     thirdProcess.testThirdH *
                     thirdProcess.testThirdI *
                     thirdProcess.testThirdJ *
                     thirdProcess.testThirdK *
                     thirdProcess.testThirdL *
                     thirdProcess.testThirdM *
                     thirdProcess.testThirdN *
                     thirdProcess.testThirdO
)

thirdProcess.endPath = cms.EndPath(thirdProcess.out)
