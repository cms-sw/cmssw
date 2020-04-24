# This process is the first step of a test that involves multiple
# processing steps. It tests the thinning collections and
# redirecting Refs, Ptrs, and RefToBases.
#
# Produce 15 thinned collections
#
#   Collection A contains Things 0-8
#   Collection B contains Things 0-3 and made from collection A
#   Collection C contains Things 4-7 and made from collection A
#
# x Collection D contains Things 10-18
#   Collection E contains Things 10-14 and made from collection D
#   Collection F contains Things 14-17 and made from collection D
#
#   Collection G contains Things 20-28
# x Collection H contains Things 20-23 and made from collection G
# x Collection I contains Things 24-27 and made from collection G
#
# x Collection J contains Things 30-38
# x Collection K contains Things 30-33 and made from collection J
# x Collection L contains Things 34-37 and made from collection J
#
# x Collection M contains Things 40-48
# x Collection N contains Things 40-43 and made from collection M
#   Collection O contains Things 44-47 and made from collection M
#
# The collections marked with an x will get deleted in the next
# processing step.
#
# The Things kept are set by creating TracksOfThings which
# reference them and using those in the selection of a
# Thinning Producer.
#
# The ThinningTestAnalyzer checks that things are working as
# they are supposed to work.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

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
process.thingProducer2 = cms.EDProducer("ThingProducer",
                                        offsetDelta = cms.int32(100),
                                        nThings = cms.int32(50)
)

process.thingProducer2alias = cms.EDAlias(
  thingProducer2 = cms.VPSet(
    cms.PSet(type = cms.string('edmtestThings'))
  )
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

process.trackOfThingsProducerDPlus = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(10, 11, 12, 13, 14, 15, 16, 17, 18, 21)
)

process.trackOfThingsProducerE = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(10, 11, 12, 13, 14)
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

process.trackOfThingsProducerD2 = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer2'),
    keysToReference = cms.vuint32(10, 11, 12, 13, 14, 15, 16, 17, 18)
)

process.trackOfThingsProducerE2 = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer2'),
    keysToReference = cms.vuint32(10, 11, 12, 13, 14)
)

process.trackOfThingsProducerF2 = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer2'),
    keysToReference = cms.vuint32(14, 15, 16, 17)
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
    inputTag = cms.InputTag('aliasM'),
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

process.testA = cms.EDAnalyzer("ThinningTestAnalyzer",
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
    fileName = cms.untracked.string('testThinningTest1.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer2_*_*',
        'drop *_thinningThingProducerM_*_*',
        'drop *_thinningThingProducerN_*_*',
        'drop *_thinningThingProducerO_*_*'
    )
)

process.out2 = cms.OutputModule("EventStreamFileWriter",
    fileName = cms.untracked.string('testThinningStreamerout.dat'),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    max_event_size = cms.untracked.int32(7000000),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer_*_*',
        'drop *_thingProducer2_*_*',
        'drop *_thinningThingProducerD_*_*',
        'drop *_thinningThingProducerH_*_*',
        'drop *_thinningThingProducerI_*_*',
        'drop *_thinningThingProducerJ_*_*',
        'drop *_thinningThingProducerK_*_*',
        'drop *_thinningThingProducerL_*_*',
        'drop *_thinningThingProducerM_*_*',
        'drop *_thinningThingProducerN_*_*',
        'drop *_thinningThingProducerO_*_*',
        'drop *_aliasM_*_*',
        'drop *_aliasN_*_*'
    )
)

process.p = cms.Path(process.thingProducer * process.thingProducer2
                                           * process.trackOfThingsProducerA
                                           * process.trackOfThingsProducerB
                                           * process.trackOfThingsProducerC
                                           * process.trackOfThingsProducerD
                                           * process.trackOfThingsProducerDPlus
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
                                           * process.trackOfThingsProducerD2
                                           * process.trackOfThingsProducerE2
                                           * process.trackOfThingsProducerF2
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
                                           * process.testA
                                           * process.testB
                                           * process.testC
                    )

process.endPath = cms.EndPath(process.out * process.out2)
