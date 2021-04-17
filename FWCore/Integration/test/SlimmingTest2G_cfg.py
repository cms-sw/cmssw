import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST2G")

process.maxEvents.input = 3

process.WhatsItESProducer = cms.ESProducer("WhatsItESProducer")

process.DoodadESSource = cms.ESSource("DoodadESSource")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring('file:testSlimmingTest1G.root')
)

# Here thinningThingProducerB leads to an association with
# default-constructed BranchID for the parent to be inserted. That
# should not lead to other failures as long as the thinning producer
# is not run.
process.rejectingFilter = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(-1)
)
process.thinningThingProducerB = cms.EDProducer("SlimmingThingProducer",
    inputTag = cms.InputTag('doesNotExist', '', 'PROD'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(50)
)

process.thinningThingProducerAB = cms.EDProducer("ThinningThingProducer",
    inputTag = cms.InputTag('thinningThingProducerA'),
    trackTag = cms.InputTag('trackOfThingsProducerB'),
    offsetToThinnedKey = cms.uint32(0),
    expectedCollectionSize = cms.uint32(20)
)

process.p = cms.Path(
    process.rejectingFilter *
    process.thinningThingProducerB
)
process.p2 = cms.Path(
    process.thinningThingProducerAB
)
