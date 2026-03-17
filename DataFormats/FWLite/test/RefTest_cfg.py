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

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('goodDataFormatsFWLite.root'),
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_thingProducer_*_*',
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
)

process.p = cms.Path(process.Thing*process.OtherThing)
process.outp = cms.EndPath(process.out*process.out2*process.out_other)
