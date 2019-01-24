import FWCore.ParameterSet.Config as cms
import sys

if len(sys.argv) > 2:
    runno = int(sys.argv[2])
else:
    runno=1

process = cms.Process("ProcessOne")

process.source= cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(runno),
    lastValue = cms.uint64(runno),
    interval = cms.uint64(1)
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

process.readSqlite = cms.EDAnalyzer("CTPPSRPAlignmentInfoReader",
                                    cms.PSet(     
        iov = cms.uint64(runno), 
        record=cms.string("RPRealAlignmentRecord")    
        )
                                    )

process.p = cms.Path(process.readSqlite)
