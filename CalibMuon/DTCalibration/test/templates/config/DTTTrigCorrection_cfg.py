import FWCore.ParameterSet.Config as cms

process = cms.Process("DTTTrigCorrection")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ''

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
    connect = cms.string('sqlite_file:ttrig.db'),
    authenticationMethod = cms.untracked.uint32(0)
)
process.es_prefer_calibDB = cms.ESPrefer('PoolDBESSource','calibDB')

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:ttrig.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    ))
)

process.DTTTrigCorrectionFirst = cms.EDAnalyzer("DTTTrigCorrectionFirst",
    debug = cms.untracked.bool(False),
    #ttrigMax = cms.untracked.double(2600.0),
    #ttrigMin = cms.untracked.double(2400.0),
    #rmsLimit = cms.untracked.double(2.)                                          
    ttrigMax = cms.untracked.double(500.0),
    ttrigMin = cms.untracked.double(200.0),
    rmsLimit = cms.untracked.double(8.)
)

process.p = cms.Path(process.DTTTrigCorrectionFirst)
