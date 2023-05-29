import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTPROD")
process.maxEvents.input = 3

process.source = cms.Source("EmptySource")

process.intProduct = cms.EDProducer("IntProducer", ivalue = cms.int32(42))

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('getbylabel_step1.root')
)

process.p = cms.Path(process.intProduct)
process.ep = cms.EndPath(process.output)
