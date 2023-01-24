import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9

process = cms.Process("CaloCellGeometryTest",Phase2C17I13M9)

process.load("Configuration.Geometry.GeometryExtended2026D92_cff")
process.load("Configuration.Geometry.GeometryExtended2026D92Reco_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCalGeomX=dict()
    process.MessageLogger.CaloGeometryBuilder=dict()

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.gtest = cms.EDAnalyzer("CaloCellGeometryTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.gtest)
