import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = -1

process.load("Configuration.Geometry.GeometryExtended2026D50_cff")

process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff")

process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdParameters_cfi")

process.Timing = cms.Service("Timing")

process.prod = cms.EDAnalyzer("GeometricTimingDetAnalyzer")

process.prod1 = cms.EDAnalyzer("MTDTopologyAnalyzer")

process.p1 = cms.Path(process.prod+process.prod1)

