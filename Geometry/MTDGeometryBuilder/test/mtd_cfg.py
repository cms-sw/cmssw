import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.INFO.limit = -1

# Choose Tracker Geometry
process.load("Configuration.Geometry.GeometryExtended2023D35_cff")

process.load("Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cfi")

process.load("Geometry.MTDNumberingBuilder.mtdTopology_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdGeometry_cfi")
process.load("Geometry.MTDGeometryBuilder.mtdParameters_cfi")
process.mtdGeometry.applyAlignment = cms.bool(False)

process.Timing = cms.Service("Timing")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.myprint = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("MTDDigiGeometryAnalyzer")

process.p1 = cms.Path(process.prod)

process.e1 = cms.EndPath(process.myprint)

