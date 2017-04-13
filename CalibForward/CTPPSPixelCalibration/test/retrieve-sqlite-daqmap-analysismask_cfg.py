import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)
#Database output service

process.load("CondCore.CondDB.CondDB_cfi")
# input database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSPixel_DAQMapping_AnalysisMask.db'
##process.CondDB.connect = 'sqlite_file:test.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('CTPPSPixelDAQMappingRcd'),
        tag = cms.string("PixelDAQMapping"),
        label = cms.untracked.string("RPix")
      ),
      cms.PSet(
        record = cms.string('CTPPSPixelAnalysisMaskRcd'),
        tag = cms.string("PixelAnalysisMask"),
        label = cms.untracked.string("RPix")
      )
    )
)



process.readSqlite = cms.EDAnalyzer("CTPPSPixelDAQMappingAnalyzer",
    cms.PSet(
        analysismaskiov = cms.uint64(1),
        daqmappingiov = cms.uint64(1),
        label  = cms.untracked.string("RPix")
    )
)

process.p = cms.Path(process.readSqlite)
