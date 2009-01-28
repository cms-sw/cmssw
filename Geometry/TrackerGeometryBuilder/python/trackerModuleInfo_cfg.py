import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
#process.print = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDFilter("ModuleInfo",
    fromDDD = cms.bool(True)
)

process.p1 = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)

