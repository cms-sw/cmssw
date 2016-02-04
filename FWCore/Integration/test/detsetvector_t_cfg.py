import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.scs = cms.EDProducer("SCSimpleProducer",
    size = cms.int32(5)
)

process.dsv1 = cms.EDProducer("DSVProducer",
    size = cms.int32(10)
)

process.a1 = cms.EDAnalyzer("SCSimpleAnalyzer")

process.a2 = cms.EDAnalyzer("DSVAnalyzer")

process.p = cms.Path(process.scs*process.dsv1*process.a1*process.a2)


