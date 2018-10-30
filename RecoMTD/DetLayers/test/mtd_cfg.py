import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Choose Tracker Geometry
process.load("Configuration.Geometry.GeometryExtended2023D24_cff")

process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cfi")

process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdGeometry_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdParameters_cfi")
process.mtdGeometry.applyAlignment = cms.bool(False)

process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")
process.load("RecoMTD.DetLayers.mtdDetLayerGeometry_cfi")

process.Timing = cms.Service("Timing")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("MTDRecoGeometryAnalyzer")

process.p1 = cms.Path(process.prod)

process.e1 = cms.EndPath(process.myprint)

