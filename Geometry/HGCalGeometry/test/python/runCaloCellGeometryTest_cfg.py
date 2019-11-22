import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloCellGeometryTest")

process.load("Configuration.Geometry.GeometryExtended2026D35_cff")
process.load("Configuration.Geometry.GeometryExtended2026D35Reco_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('HGCalGeom')
    process.MessageLogger.categories.append('CaloGeometryBuilder')


process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.gtest = cms.EDAnalyzer("CaloCellGeometryTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.gtest)
