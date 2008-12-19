import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.source = cms.Source("EmptySource",
    maxEvents = cms.untracked.int32(1)
)

process.print = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDFilter("TrackerMapTool")

process.p1 = cms.Path(process.prod)
process.ep = cms.EndPath(process.print)


