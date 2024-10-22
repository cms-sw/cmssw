import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPROD")
process.maxEvents.input = 11

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(6),
    numberEventsInLuminosityBlock = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(561),
    numberEventsInRun = cms.untracked.uint32(7)
)

process.parentIntProduct = cms.EDProducer("edmtest::TransientIntParentProducer", ivalue = cms.int32(1))

process.intProduct = cms.EDProducer("edmtest::IntProducerFromTransientParent",
    src = cms.InputTag("parentIntProduct")
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('noparentdict_step1.root')
)

process.p = cms.Path(process.parentIntProduct+process.intProduct)
process.ep = cms.EndPath(process.output)
