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

process.anotherThingProducer = cms.EDProducer("DetSetVectorThingProducer",
    offsetDelta = cms.int32(100),
    nThings = cms.int32(50),
    detSets = cms.vint32(1,2,3)
)

process.thirdThingProducer = cms.EDProducer("DetSetVectorThingProducer",
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

process.trackOfAnotherThingsProducerA = cms.EDProducer("TrackOfDSVThingsProducer",
    inputTag = cms.InputTag('anotherThingProducer'),
    keysToReference = cms.vuint32(
        list(range(0,5)) +
        list(range(50,55)) +
        list(range(100,105))
    ),
    nTracks = cms.uint32(4*3)
)

process.trackOfThirdThingsProducerA = cms.EDProducer("TrackOfDSVThingsProducer",
    inputTag = cms.InputTag('thirdThingProducer'),
    keysToReference = cms.vuint32(
        list(range(0,3)) +
        list(range(50,53)) +
        list(range(100,103))
    ),
    nTracks = cms.uint32(2*3)
)

process.thinningThingProducerA = cms.EDProducer("ThinningDSVThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
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

process.thinningAnotherThingProducerA = cms.EDProducer("ThinningDSVThingProducer",
    inputTag = cms.InputTag('anotherThingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    expectedDetSets = cms.uint32(3),
    expectedDetSetSize = cms.uint32(50),
    thinnedRefSetIgnoreInvalidParentRef = cms.bool(True),
)

process.thinningAnotherThingProducerA2 = cms.EDProducer("ThinningDSVThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfAnotherThingsProducerA'),
    expectedDetSets = cms.uint32(3),
    expectedDetSetSize = cms.uint32(50),
    thinnedRefSetIgnoreInvalidParentRef = cms.bool(True),
)

process.thinningAnotherThingProducerA3 = cms.EDProducer("ThinningDSVThingProducer",
    inputTag = cms.InputTag('anotherThingProducer'),
    trackTag = cms.InputTag('trackOfThirdThingsProducerA'),
    expectedDetSets = cms.uint32(3),
    expectedDetSetSize = cms.uint32(50),
    thinnedRefSetIgnoreInvalidParentRef = cms.bool(True),
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

process.anotherTestA = cms.EDAnalyzer("ThinningDSVTestAnalyzer",
    parentTag = cms.InputTag('anotherThingProducer'),
    thinnedTag = cms.InputTag('thinningAnotherThingProducerA'),
    associationTag = cms.InputTag('thinningAnotherThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerA'),
    expectedParentContent = cms.VPSet(),
    expectedThinnedContent = cms.VPSet(),
    expectedIndexesIntoParent = cms.vuint32(),
    expectedNumberOfTracks = cms.uint32(8*3),
    expectedValues = cms.vint32()
)

process.anotherTestA2 = cms.EDAnalyzer("ThinningDSVTestAnalyzer",
    parentTag = cms.InputTag('thingProducer'),
    thinnedTag = cms.InputTag('thinningAnotherThingProducerA2'),
    associationTag = cms.InputTag('thinningAnotherThingProducerA2'),
    trackTag = cms.InputTag('trackOfAnotherThingsProducerA'),
    expectedParentContent = cms.VPSet(),
    expectedThinnedContent = cms.VPSet(),
    expectedIndexesIntoParent = cms.vuint32(),
    expectedNumberOfTracks = cms.uint32(4*3),
    expectedValues = cms.vint32()
)

process.anotherTestA3 = cms.EDAnalyzer("ThinningDSVTestAnalyzer",
    parentTag = cms.InputTag('anotherThingProducer'),
    thinnedTag = cms.InputTag('thinningAnotherThingProducerA3'),
    associationTag = cms.InputTag('thinningAnotherThingProducerA3'),
    trackTag = cms.InputTag('trackOfThirdThingsProducerA'),
    expectedParentContent = cms.VPSet(),
    expectedThinnedContent = cms.VPSet(),
    expectedIndexesIntoParent = cms.vuint32(),
    expectedNumberOfTracks = cms.uint32(2*3),
    expectedValues = cms.vint32()
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
    * process.anotherThingProducer
    * process.thirdThingProducer
    * process.trackOfThingsProducerA
    * process.trackOfAnotherThingsProducerA
    * process.trackOfThirdThingsProducerA
    * process.thinningThingProducerA
    * process.slimmingThingProducerA
    * process.thinningAnotherThingProducerA
    * process.thinningAnotherThingProducerA2
    * process.thinningAnotherThingProducerA3
    * process.testA
    * process.slimmingTestA
    * process.anotherTestA
    * process.anotherTestA2
    * process.anotherTestA3
)

process.ep = cms.EndPath(
    process.out
)
