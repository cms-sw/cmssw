import FWCore.ParameterSet.Config as cms

process = cms.Process("p")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.source = cms.Source("EmptySource")

process.m1 = cms.EDAnalyzer("TestFailuresAnalyzer",
    whichFailure = cms.int32(4)
)

process.p1 = cms.Path(process.m1)
