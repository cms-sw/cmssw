import FWCore.ParameterSet.Config as cms

process = cms.Process("TESTMISSING")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.Thing = cms.EDProducer("ThingProducer",
    noPut = cms.untracked.bool(True)
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:PoolMissingTest.root')
)

process.source = cms.Source("EmptySource")

process.p = cms.Path(process.Thing)
process.ep = cms.EndPath(process.output)


