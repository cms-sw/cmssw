import FWCore.ParameterSet.Config as cms

process = cms.Process("EARLIER")

process.source = cms.Source("EmptySource")
process.maxEvents.input = 3

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("testOutputModuleSelectEventsMissingPath.root")
)

process.intprod = cms.EDProducer("IntProducer", ivalue=cms.int32(3))

process.p = cms.Path(process.intprod)

process.ep3 = cms.EndPath(process.out)

