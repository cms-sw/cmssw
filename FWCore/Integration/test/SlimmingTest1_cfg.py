# This process is the first step of a test that involves multiple
# processing steps. It tests the thinning and slimming collections and
# redirecting Refs, Ptrs, and RefToBases.
#
# From thingProducer collection (50 elements)
# - A: elements 0-19
# - B: elements 0-10
# - C: elements 11-19
# - D: elements 0-5
# - E: elements 6-10
# - F: elements 6-8
# - G: elements 9-10
# - H: elements 0-19
# - I: elements 11-15
#
# thingProducer (0-49)
# |\- thinningThingProducerA (0-19)
# |   \- thinningThingProducerAB (0-10)
# |      |\- thinningThingProducerABE (6-10)
# |      |   \- thinningThingProducerABEF (6-8) slimmed (in job reading file F)
# |      |
# |      |\- thinningThingProducerABF (6-8)
# |      |
# |      \-- thinningThingProducerABG (9-10)
# |
# |\- thinningThingProducerB (0-10) slimmed
# |   |\- thinningThingProducerBD (0-5)
# |   |
# |   |\- thinningThingProducerBE (6-10) slimmed
# |   |   |\- thinningThingProducerBEF (6-8)
# |   |   |
# |   |   \-- thinningThingProducerBEG (9-10)
# |   |
# |   \- thinningThingProducerBF (6-8)
# |
# |\- thinningThingProducerC (11-19)
# |   \- thinnindThingProducerCI (11-15)
# |
# \-- thinningThingProducerH (0-19), no product because of event filtering
#
# cases to be tested
# A:  everything can be put in one file, when reading thinned are read
# B: file including ABF, ABG, BF, BEG
#   - ABF is preferred over BF for elements 6-8
#   - ABG is preferred over BEG for elements 9-10
#   - then drop ABF, ABG, and can get BF but not BEG
#   - then drop BF, can get BEG
# C: file including BF, BEF
#   - BF is preferred of BEF
#   - then drop BF, and get BEF
# D: file including ABE, BD
#   - ABE is preferred over BD for elements 0-5 (leading to product not found)
#   - drop ABE, then BD works
# E: file including BD, BEF
#   - BD is preferred over BEF for elements 6-8 (leading to product not found)
#   - drop BD, then BEF works
# F: file including ABE
#   - slim ABE further to ABEF
#   - then try to merge with B file with BEG, should fail
# G: file including A
#   - try to thin from thingProducer, should fail
# H: file including H, B
#   - non-existing H is preferred, all Refs are Null
# I: file including B, CI
#   - CI is preferred over B for elements 0-10 (leading to product not fond)
#   - drop CU, then B works

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

process.maxEvents.input = 3

process.source = cms.Source("EmptySource")

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.thingProducer = cms.EDProducer("ThingProducer",
                                       offsetDelta = cms.int32(100),
                                       nThings = cms.int32(50)
)

process.trackOfThingsProducerA = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(range(0, 20))
)

process.trackOfThingsProducerB = process.trackOfThingsProducerA.clone(keysToReference = range(0,11))
process.trackOfThingsProducerC = process.trackOfThingsProducerA.clone(keysToReference = range(11,20))
process.trackOfThingsProducerD = process.trackOfThingsProducerA.clone(keysToReference = range(0,6))
process.trackOfThingsProducerE = process.trackOfThingsProducerA.clone(keysToReference = range(6,11))
process.trackOfThingsProducerF = process.trackOfThingsProducerA.clone(keysToReference = range(6,9))
process.trackOfThingsProducerG = process.trackOfThingsProducerA.clone(keysToReference = range(9,11))
process.trackOfThingsProducerH = process.trackOfThingsProducerA.clone()
process.trackOfThingsProducerI = process.trackOfThingsProducerA.clone(keysToReference = range(11,16))

process.thinningThingProducerA = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)
process.thinningThingProducerAB = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(20)
)
process.thinningThingProducerABE = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerAB'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(11)
)
process.thinningThingProducerABF = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerAB'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(11)
)
process.thinningThingProducerABG = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerAB'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(11)
)

process.thinningThingProducerB = cms.EDProducer("SlimmingThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)
process.thinningThingProducerBD = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(11),
    slimmedValueFactor = cms.int32(10)
)
process.thinningThingProducerBE = cms.EDProducer("SlimmingThingProducer",
    inputTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(11),
    slimmedValueFactor = cms.int32(10)
)
process.thinningThingProducerBEF = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerBE'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    offsetToThinnedKey = cms.uint32(6),
    expectedCollectionSize = cms.uint32(5),
    slimmedValueFactor = cms.int32(100)
)
process.thinningThingProducerBEG = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerBE'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    offsetToThinnedKey = cms.uint32(6),
    expectedCollectionSize = cms.uint32(5),
    slimmedValueFactor = cms.int32(100)
)
process.thinningThingProducerBF = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(11),
    slimmedValueFactor = cms.int32(10)
)

process.thinningThingProducerC = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerC'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)
process.thinningThingProducerCI = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerC'),
    trackTag = cms.InputTag('trackOfThingsProducerI'),
    offsetToThinnedKey = cms.uint32(11),
    offsetToValue = cms.uint32(11),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerH = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerH'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)
process.rejectingFilter = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(-1)
)

process.testA = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerA'),
    associationTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    expectedParentContent = cms.vint32(range(0,50)),
    expectedThinnedContent = cms.vint32(range(0,20)),
    expectedIndexesIntoParent = cms.vuint32(range(0,20)),
    expectedValues = cms.vint32(range(0,20)),
)

process.testABF = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerAB'),
    thinnedTag = cms.InputTag('thinningThingProducerABF'),
    associationTag = cms.InputTag('thinningThingProducerABF'),
    trackTag = cms.InputTag('trackOfThingsProducerF'),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,9)),
    expectedIndexesIntoParent = cms.vuint32(range(6,9)),
    expectedValues = cms.vint32(range(6,9)),
)

process.testB = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerB'),
    associationTag = cms.InputTag('thinningThingProducerB'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    thinnedSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(0,50)),
    expectedThinnedContent = cms.vint32(range(0,11)),
    expectedIndexesIntoParent = cms.vuint32(range(0,11)),
    expectedValues = cms.vint32(range(0,11)),
)

process.testBD = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBD'),
    associationTag = cms.InputTag('thinningThingProducerBD'),
    trackTag = cms.InputTag('trackOfThingsProducerD'),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(1),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(0,6)),
    expectedIndexesIntoParent = cms.vuint32(range(0,6)),
    expectedValues = cms.vint32(range(0,6)),
)

process.testBE = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerB'),
    thinnedTag = cms.InputTag('thinningThingProducerBE'),
    associationTag = cms.InputTag('thinningThingProducerBE'),
    trackTag = cms.InputTag('trackOfThingsProducerE'),
    parentSlimmedCount = cms.int32(1),
    thinnedSlimmedCount = cms.int32(2),
    expectedParentContent = cms.vint32(range(0,11)),
    expectedThinnedContent = cms.vint32(range(6,11)),
    expectedIndexesIntoParent = cms.vuint32(range(6,11)),
    expectedValues = cms.vint32(range(6,11)),
)

process.testBEG = cms.EDAnalyzer("ThinningTestAnalyzer",
    parentTag = cms.InputTag('thinningThingProducerBE'),
    thinnedTag = cms.InputTag('thinningThingProducerBEG'),
    associationTag = cms.InputTag('thinningThingProducerBEG'),
    trackTag = cms.InputTag('trackOfThingsProducerG'),
    parentSlimmedCount = cms.int32(2),
    thinnedSlimmedCount = cms.int32(2),
    expectedParentContent = cms.vint32(range(6,11)),
    expectedThinnedContent = cms.vint32(range(9,11)),
    expectedIndexesIntoParent = cms.vuint32(range(3,5)),
    expectedValues = cms.vint32(range(9,11)),
)



process.outA = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1A.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer_*_*',
    )
)

process.outB = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1B.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerF_*_*',
        'keep *_trackOfThingsProducerG_*_*',
        'keep *_thinningThingProducerABF_*_*',
        'keep *_thinningThingProducerABG_*_*',
        'keep *_thinningThingProducerBF_*_*',
        'keep *_thinningThingProducerBEG_*_*',
    )
)

process.outC = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1C.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerF_*_*',
        'keep *_thinningThingProducerBF_*_*',
        'keep *_thinningThingProducerBEF_*_*',
    )
)

process.outD = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1D.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerD_*_*',
        'keep *_trackOfThingsProducerE_*_*',
        'keep *_thinningThingProducerABE_*_*',
        'keep *_thinningThingProducerBD_*_*',
    )
)

process.outE = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1E.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerD_*_*',
        'keep *_trackOfThingsProducerF_*_*',
        'keep *_thinningThingProducerBD_*_*',
        'keep *_thinningThingProducerBEF_*_*',
    )
)

process.outF = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1F.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerF_*_*',
        'keep *_thinningThingProducerABE_*_*',
    )
)

process.outG = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1G.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerB_*_*',
        'keep *_thinningThingProducerA_*_*',
    )
)

process.outH = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1H.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_thinningThingProducerB_*_*',
        'keep *_trackOfThingsProducerH_*_*',
        'keep *_thinningThingProducerH_*_*',
    )
)

process.outI = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSlimmingTest1I.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerB_*_*',
        'keep *_trackOfThingsProducerI_*_*',
        'keep *_thinningThingProducerB_*_*',
        'keep *_thinningThingProducerCI_*_*',
    )
)


process.p = cms.Path(
    process.thingProducer
    * process.trackOfThingsProducerA
    * process.trackOfThingsProducerB
    * process.trackOfThingsProducerC
    * process.trackOfThingsProducerD
    * process.trackOfThingsProducerE
    * process.trackOfThingsProducerF
    * process.trackOfThingsProducerG
    * process.trackOfThingsProducerI
    * process.thinningThingProducerA
    * process.thinningThingProducerAB
    * process.thinningThingProducerABE
    * process.thinningThingProducerABF
    * process.thinningThingProducerABG
    * process.thinningThingProducerB
    * process.thinningThingProducerBD
    * process.thinningThingProducerBE
    * process.thinningThingProducerBEF
    * process.thinningThingProducerBEG
    * process.thinningThingProducerBF
    * process.thinningThingProducerC
    * process.thinningThingProducerCI
    * process.testA
    * process.testABF
    * process.testB
    * process.testBD
    * process.testBE
    * process.testBEG
)
process.p2 = cms.Path(
    process.thingProducer
    * process.trackOfThingsProducerH
    * process.rejectingFilter
    * process.thinningThingProducerH
)

process.ep = cms.EndPath(
    process.outA
    * process.outB
    * process.outC
    * process.outD
    * process.outE
    * process.outF
    * process.outG
    * process.outH
    * process.outI
)
