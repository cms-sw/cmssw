import FWCore.ParameterSet.Config as cms

process = cms.Process("writeCTPPSBeamParameters")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:CTPPSBeamParameters.db'

# A data source must always be defined. We don't need it, so here's a dummy one.
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    # timetype = cms.string('lumiid'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# We define the output service.
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    # timetype = cms.untracked.string('runnumber'),
    # timetype = cms.untracked.string('timestamp'),
    timetype = cms.untracked.string('lumiid'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSBeamParametersRcd'),
        tag = cms.string('CTPPSBeamParameters_v1')
    ))
)

process.parameters_maker = cms.EDAnalyzer("WriteCTPPSBeamParameters",
    record = cms.string('CTPPSBeamParametersRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
        IOVRun=cms.untracked.uint32(1)
    )
)

process.path = cms.Path(process.parameters_maker)
