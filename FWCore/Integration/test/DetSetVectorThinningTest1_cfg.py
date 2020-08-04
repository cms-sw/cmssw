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

process.thingProducer = cms.EDProducer(
    "DetSetVectorThingProducer",
    offsetDelta = cms.int32(100),
    nThings = cms.int32(50),
    detSets = cms.vint32(1,2,3)
)

process.trackOfThingsProducerA = cms.EDProducer("TrackOfDSVThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(
        list(range(0,9)) +
        list(range(50,59)) +
        list(range(100,109))
    ),
    nTracks = cms.uint32(8*3)
)

process.thinningThingProducerA = cms.EDProducer("ThinningDSVThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    offsetToThinnedKey = cms.uint32(0),
    expectedDetSets = cms.uint32(3),
    expectedDetSetSize = cms.uint32(50),
)

process.slimmingThingProducerA = cms.EDProducer("SlimmingDSVThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    offsetToThinnedKey = cms.uint32(0),
    expectedDetSets = cms.uint32(3),
    expectedDetSetSize = cms.uint32(50),
)

process.testA = cms.EDAnalyzer("ThinningDSVTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningThingProducerA'),
    associationTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
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

process.slimmingTestA = cms.EDAnalyzer("ThinningDSVTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('slimmingThingProducerA'),
    associationTag = cms.InputTag('slimmingThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    thinnedSlimmedCount = cms.int32(1),
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

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testDetSetVectorThinningTest1.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_trackOfThingsProducerA_*_*',
        'keep *_slimmingThingProducerA_*_*',
    )
)


process.p = cms.Path(
    process.thingProducer
    * process.trackOfThingsProducerA
    * process.thinningThingProducerA
    * process.slimmingThingProducerA
    * process.testA
    * process.slimmingTestA
)

process.ep = cms.EndPath(
    process.out
)
