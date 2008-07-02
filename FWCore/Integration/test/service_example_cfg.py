import FWCore.ParameterSet.Config as cms

process = cms.Process("test")

process.load("FWCore.Framework.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)

process.source = cms.Source("EmptySource")

process.tester = cms.EDAnalyzer("UseValueExampleAnalyzer")

process.ValueExample = cms.Service("ValueExample",
    value = cms.int32(2)
)

process.p = cms.Path(process.tester)


