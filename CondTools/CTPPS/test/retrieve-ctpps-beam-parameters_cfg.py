import FWCore.ParameterSet.Config as cms

process = cms.Process("retrieveCTPPSBeamParametersTest")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# input database (in this case the local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSBeamParameters.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
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
