import FWCore.ParameterSet.Config as cms
process = cms.Process("GeometryTest")

# minimum of logs
process.load("Configuration.TotemCommon.LoggerMin_cfi")

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRP_cfi")

# no events to process
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.GeomInfo = cms.EDAnalyzer("GeometryTestModule")

process.p = cms.Path(
    process.GeomInfo
)
