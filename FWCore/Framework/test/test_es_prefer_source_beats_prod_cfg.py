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

process.bad = cms.ESProducer("LoadableDummyProvider",
    value = cms.untracked.int32(0)
)

process.good = cms.ESSource("LoadableDummyESSource",
    value = cms.untracked.int32(5)
)

process.prefer("good")
process.p1 = cms.Path(process.m)
# foo bar baz
# sgmldX1nGevk0
# MRrogjrdrCdnW
