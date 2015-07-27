import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Geometry.HcalCommonData.hcalParameters_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalParametersAnalyzer")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
