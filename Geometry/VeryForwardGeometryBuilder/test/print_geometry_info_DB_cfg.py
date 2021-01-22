import FWCore.ParameterSet.Config as cms
process = cms.Process("GeometryInfo")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")
#process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi")

# no events to process
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
# input database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSRPAlignment.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('RPMisalignedAlignmentRecord'),
        tag = cms.string("CTPPSRPAlignment_misaligned")
      )
    )
)

process.ctppsGeometryInfo = cms.EDAnalyzer("CTPPSGeometryInfo",
    geometryType = cms.untracked.string("misaligned"),
    printRPInfo = cms.untracked.bool(True),
    printSensorInfo = cms.untracked.bool(True)
)

process.p = cms.Path(
    process.ctppsGeometryInfo
)
