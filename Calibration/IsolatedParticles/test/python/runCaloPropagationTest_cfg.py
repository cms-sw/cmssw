import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load('Geometry.HcalCommonData.testGeometry17bXML_cfi')
process.load('Geometry.HcalCommonData.hcalDDConstants_cff')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.hpa = cms.EDAnalyzer("HcalRecNumberingTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
