import FWCore.ParameterSet.Config as cms
process = cms.Process("GeometryInfo")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")

# load alignment correction
process.load("Geometry.VeryForwardGeometryBuilder.ctppsIncludeAlignmentsFromXML_cfi")
process.ctppsIncludeAlignmentsFromXML.RealFiles = cms.vstring("Geometry/VeryForwardGeometryBuilder/test/sample_alignment_corrections.xml")

# no events to process
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.ctppsGeometryInfo = cms.EDAnalyzer("CTPPSGeometryInfo",
    geometryType = cms.untracked.string("real"),
    printRPInfo = cms.untracked.bool(True),
    printSensorInfo = cms.untracked.bool(True)
)

process.p = cms.Path(
    process.ctppsGeometryInfo
)
