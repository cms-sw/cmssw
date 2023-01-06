import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerTopology_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.load("Alignment.OfflineValidation.TrackerGeometryCompare_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("TrackerMapTool")

process.p1 = cms.Path(process.prod)
