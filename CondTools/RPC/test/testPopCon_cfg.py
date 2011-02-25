import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'sqlite_file:pop_test.db'
process.CondDBCommon.connect = 'sqlite_file:RPCStripNoise_COSMIC09_mc.db'
#process.CondDBCommon.connect = 'sqlite_file:BadChambers_5ME42.db'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('RPCStripNoisesRcd'),
#        tag = cms.string('RPCStripNoises_v2_mc')
#        tag = cms.string('RPCStripNoises_v1_mc')
        tag = cms.string('RPCStripNoise_COSMIC09_mc')
    ))
)

process.Test1 = cms.EDAnalyzer("MyRPCDBPopConAnalyzer",
##    debug = cms.bool(True),
#    debug = cms.bool(False),
#    record = cms.string('RPCStripNoisesRcd'),
    record = cms.string('RPCStripNoisesRcd'),
                               loggingOn= cms.untracked.bool(True),
                               IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
                               SinceAppendMode=cms.bool(True),
#
#    SinceAppendMode = cms.bool(False),
#    SinceAppendMode = cms.bool(True),
#    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        effmapfile = cms.FileInPath('CondTools/RPC/data/RPCDetId_Eff.dat'),
        noisemapfile = cms.FileInPath('CondTools/RPC/data/RPCDetId_Noise.dat'),
        clsmapfile = cms.FileInPath('CondTools/RPC/data/ClSizeTot.dat'),
        firstSince = cms.untracked.int32(1),
#        firstSince = cms.untracked.int32(10),
#        tag = cms.string('RPCStripNoises_v2_mc'),
        tag = cms.string('RPCStripNoises_v1_mc'),
        timingMap = cms.FileInPath('CondTools/RPC/data/RPCTiming.dat')
    )
)

process.p = cms.Path(process.Test1)
#process.CondDBCommon.connect = 'sqlite_file:pop_test.db'
#process.CondDBCommon.DBParameters.authenticationPath = '.'
#process.CondDBCommon.DBParameters.messageLevel = 1


