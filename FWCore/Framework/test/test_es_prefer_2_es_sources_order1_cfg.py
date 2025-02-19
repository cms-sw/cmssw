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

process.good = cms.ESSource("LoadableDummyESSource",
    startRunForIOV = cms.untracked.uint32(1),
    value = cms.untracked.int32(5)
)

process.bad = cms.ESSource("EmptyESSource",
    recordName = cms.string('DummyRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(10)
)

process.prefer("good")
process.p1 = cms.Path(process.m)
