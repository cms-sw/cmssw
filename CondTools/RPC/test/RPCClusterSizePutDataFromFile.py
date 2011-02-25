import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
#    lastValue = cms.uint64(1),                            
    timetype = cms.string('runnumber'),
#    firstValue = cms.uint64(11),
    firstValue = cms.uint64(1),                    
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('runnumber'),
#    connect = cms.string('sqlite_file:iov_test.db'),
    connect = cms.string('sqlite_file:RPCClusterSize_mc.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('RPCClusterSizeRcd'),
#        tag = cms.string('RPCClusterSizeRcd')
        tag = cms.string('RPCClusterSize_mc')
    ))
)

##process.mytest = cms.EDAnalyzer('RiovTest')
process.mytest = cms.EDAnalyzer('CondToolsRPCP',
##process.mytest = cms.EDAnalyzer('RPCDBClsPerformanceHandler',
                                record = cms.string('RPCClusterSizeRcd'),
                                clsidmapfile =
                                cms.FileInPath('CalibMuon/RPCCalibration/data/RPCDetId_ClSizeTot.dat')
                                )

process.p = cms.Path(process.mytest)



