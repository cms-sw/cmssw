import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.statistics.append("debugs")
process.MessageLogger.categories.append("MTDLayerDump")
process.MessageLogger.categories.append("MTDDetLayers")
process.MessageLogger.destinations.append('debugs')
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.debugs = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'),
    INFO= cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    MTDLayerDump = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
    MTDDetLayers = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
)

# Choose Tracker Geometry
process.load("Configuration.Geometry.GeometryExtended2026D50_cff")

process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff")

process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdParameters_cff")

process.load("Geometry.MTDGeometryBuilder.mtdGeometry_cfi")
process.mtdGeometry.applyAlignment = cms.bool(False)

process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")
process.load("RecoMTD.DetLayers.mtdDetLayerGeometry_cfi")

process.Timing = cms.Service("Timing")

process.prod = cms.EDAnalyzer("MTDRecoGeometryAnalyzer")

process.p1 = cms.Path(process.prod)
