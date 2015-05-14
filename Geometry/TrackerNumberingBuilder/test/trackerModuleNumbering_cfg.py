import FWCore.ParameterSet.Config as cms

process = cms.Process("NumberingTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Choose Tracker Geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.print = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("ModuleNumbering")

process.p1 = cms.Path(process.prod)
process.ep = cms.EndPath(process.print)


