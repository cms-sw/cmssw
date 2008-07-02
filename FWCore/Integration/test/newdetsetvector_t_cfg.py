import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)
process.source = cms.Source("EmptySource")

process.dsv1 = cms.EDProducer("DSVProducer",
    size = cms.int32(10)
)

process.dsv2 = cms.EDProducer("DSTVProducer",
    size = cms.int32(10)
)

process.a2 = cms.EDAnalyzer("DSVAnalyzer")

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testDSTV.root')
)

process.p = cms.Path(process.dsv1 * process.dsv2 * process.a2)
process.e = cms.EndPath(process.out)
