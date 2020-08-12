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
#process.load("CondFormats.CTPPSReadoutObjects.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(
    "Geometry/VeryForwardGeometryBuilder/test/alignment_file_1.xml",
    "Geometry/VeryForwardGeometryBuilder/test/alignment_file_2.xml",
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
