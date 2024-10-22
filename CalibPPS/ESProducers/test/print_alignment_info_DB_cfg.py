import FWCore.ParameterSet.Config as cms
process = cms.Process("GeometryInfo")

import sys

if len(sys.argv) >1:
    runno = int(sys.argv[1])
else:
    runno = 1

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

# load alignment correction
#process.load("Geometry.VeryForwardGeometryBuilder.ctppsIncludeAlignmentsFromXML_cfi")
#process.ctppsIncludeAlignmentsFromXML.RealFiles = cms.vstring("Geometry/VeryForwardGeometryBuilder/test/sample_alignment_corrections.xml")

# no events to process
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(runno),
    lastValue = cms.uint64(runno),
    interval = cms.uint64(1)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#Database output service
process.load("CondCore.CondDB.CondDB_cfi")
# input database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSRPRealAlignment_table.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('RPRealAlignmentRecord'),
        tag = cms.string("CTPPSRPAlignment_real_table")
      )
    )
)


process.ctppsAlignmentInfo = cms.EDAnalyzer("CTPPSAlignmentInfo",
    alignmentType = cms.untracked.string("real"),
)

process.p = cms.Path(
    process.ctppsAlignmentInfo
)
