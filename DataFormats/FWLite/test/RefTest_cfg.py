# Configuration file for RefTest_t   

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.Thing = cms.EDProducer("ThingProducer",
    offsetDelta = cms.int32(1)
)

process.OtherThing = cms.EDProducer("OtherThingProducer")

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

process.trackOfThingsProducerDMinus = cms.EDProducer("TrackOfThingsProducer",
    inputTag = cms.InputTag('thingProducer'),
    keysToReference = cms.vuint32(10, 11, 12, 13, 14, 15, 16, 17)
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

process.thinningThingProducerN = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerM'),
    trackTag = cms.InputTag('trackOfThingsProducerN'),
    offsetToThinnedKey = cms.uint32(40),
    expectedCollectionSize = cms.uint32(9)
)

process.thinningThingProducerO = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerM'),
    trackTag = cms.InputTag('trackOfThingsProducerO'),
    offsetToThinnedKey = cms.uint32(40),
    expectedCollectionSize = cms.uint32(9)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('goodDataFormatsFWLite.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer_*_*',
        'drop *_thinningThingProducerD_*_*',
        'drop *_thinningThingProducerH_*_*',
        'drop *_thinningThingProducerI_*_*',
        'drop *_thinningThingProducerJ_*_*',
        'drop *_thinningThingProducerK_*_*',
        'drop *_thinningThingProducerL_*_*',
        'drop *_thinningThingProducerM_*_*',
        'drop *_thinningThingProducerN_*_*',
    )
)

process.out2 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('good2DataFormatsFWLite.root')
)

process.out_other = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmtestOtherThings_*_*_*', 
        'keep *_TriggerResults_*_*'),
    fileName = cms.untracked.string('other_onlyDataFormatsFWLite.root')
)

process.thinningTestPath = cms.Path(process.thingProducer
                                    * process.trackOfThingsProducerA
                                    * process.trackOfThingsProducerB
                                    * process.trackOfThingsProducerC
                                    * process.trackOfThingsProducerD
                                    * process.trackOfThingsProducerDMinus
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
)

process.p = cms.Path(process.Thing*process.OtherThing)
process.outp = cms.EndPath(process.out*process.out2*process.out_other)
