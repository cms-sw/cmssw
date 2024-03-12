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
    startRunForIOV = cms.untracked.uint32(10),
    value = cms.untracked.int32(5)
)

process.bad = cms.ESSource("EmptyESSource",
    recordName = cms.string('DummyRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.p1 = cms.Path(process.m)
# foo bar baz
# QdBm02Jhnmbg1
# IZTcbQwKLPJO7
