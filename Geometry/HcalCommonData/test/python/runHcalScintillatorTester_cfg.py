import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HCalGeom')

process.hpa = cms.EDAnalyzer("HcalScintillatorTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.hpa)
