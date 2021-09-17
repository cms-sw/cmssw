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

process.trackOfThingsProducerB = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(0, 1, 2, 3)
)

process.trackOfThingsProducerC = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(4, 5, 6, 7)
)

process.slimmingThingProducerB = cms.EDProducer("SlimmingThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.slimmingThingProducerC = cms.EDProducer("SlimmingThingProducer",
    inputTag = cms.InputTag('thingProducer'),
    trackTag = cms.InputTag('trackOfThingsProducerC'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.p = cms.Path(
    process.thingProducer *
    process.trackOfThingsProducerB *
    process.trackOfThingsProducerC *
    process.slimmingThingProducerB *
    process.slimmingThingProducerC
)
