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
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi")

# load alignment correction
process.load("CondFormats.PPSObjects.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(
    "Geometry/VeryForwardGeometryBuilder/test/CTPPSIncludeAlignmentCorrectionsXML/alignment_file_1.xml",
    "Geometry/VeryForwardGeometryBuilder/test/CTPPSIncludeAlignmentCorrectionsXML/alignment_file_2.xml",
)
process.ctppsRPAlignmentCorrectionsDataESSourceXML.verbosity = 1

# no events to process
process.source = cms.Source("EmptySource",
  firstRun = cms.untracked.uint32(123),
  firstLuminosityBlock = cms.untracked.uint32(1),
  firstEvent = cms.untracked.uint32(1),
  numberEventsInLuminosityBlock = cms.untracked.uint32(3),
  numberEventsInRun = cms.untracked.uint32(30)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

process.ctppsGeometryInfo = cms.EDAnalyzer("CTPPSGeometryInfo",
    geometryType = cms.untracked.string("real"),
    printRPInfo = cms.untracked.bool(False),
    printSensorInfo = cms.untracked.bool(False)
)

process.p = cms.Path(
    process.ctppsGeometryInfo
)
