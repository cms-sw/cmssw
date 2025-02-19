import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.source = cms.Source("EmptySource")

process.m = cms.EDAnalyzer("TestESDummyDataAnalyzer",
    expected = cms.int32(5)
)

process.good = cms.ESProducer("LoadableDummyProvider",
    value = cms.untracked.int32(5)
)

process.bad = cms.ESSource("LoadableDummyESSource",
    value = cms.untracked.int32(0)
)

process.p1 = cms.Path(process.m)
