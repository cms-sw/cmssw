import FWCore.ParameterSet.Config as cms

process = cms.Process("BuildPPSGeometry")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:PPSGeometry_2017.db'

# We define the output service.
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('VeryForwardIdealGeometryRecord'),
        tag = cms.string('PPSGeometry_test')
    ))
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('ppsGeometryBuilder')
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'))

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

# geometry
process.load("Geometry.VeryForwardGeometry.dd4hep.geometryRPFromDD_2021_cfi")

# DB writer
process.ppsGeometryBuilder = cms.EDAnalyzer("PPSGeometryBuilder",
    fromDD4hep = cms.untracked.bool(True),
    isRun2 = cms.untracked.bool(False),
    compactViewTag = cms.untracked.string('XMLIdealGeometryESSource_CTPPS')
)

process.p = cms.Path(
    process.ppsGeometryBuilder
)
