import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloCellGeometryTest")

process.load("Configuration.Geometry.GeometryExtended2023D19_cff")
process.load("Configuration.Geometry.GeometryExtended2023D19Reco_cff")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.gtest = cms.EDAnalyzer("CaloCellGeometryTester")

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.gtest)
