import FWCore.ParameterSet.Config as cms

process = cms.Process("retrieveCTPPSBeamParametersTest")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# input database (in this case the local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSBeamParameters.db'
# WORK_DIR = "/afs/cern.ch/work/w/wcarvalh/ctpps/db/CMSSW_10_4_0_pre2/src"
# process.CondDB.connect = 'sqlite_file:/afs/cern.ch/work/w/wcarvalh/ctpps/db/CMSSW_10_2_0_pre3/src/CondTools/CTPPS/test_output.db'

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSBeamParametersRcd'),
        tag = cms.string("CTPPSBeamParameters_v1")
    )),
)

process.get = cms.EDAnalyzer("RetrieveCTPPSBeamParameters",
    verbose = cms.untracked.bool(True)
)

# A data source must always be defined. We don't need it, so here's a dummy one.
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(3),
    lastValue = cms.uint64(3),
    interval = cms.uint64(1)
    #timetype = cms.string('timestamp'),
    #firstValue = cms.uint64(6542600000000000000),
    #lastValue = cms.uint64(6542700000000000000),
    #interval = cms.uint64(100000000000000)
    # timetype = cms.string('lumiid'),
)

process.path = cms.Path(process.get)
