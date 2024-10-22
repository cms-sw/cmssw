import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_39X_PRESHOWER'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:ESTimeSampleWeights_HG.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(
    # weights
    record = cms.string('ESTimeSampleWeightsRcd'),
    tag = cms.string('ESTimeSampleWeights_HG')

    )))

process.ecalModule = cms.EDAnalyzer("StoreESCondition",
                                    logfile = cms.string('./logfile.log'),
                                    gain = cms.uint32(2),
                                    toPut = cms.VPSet(cms.PSet(

    # weights
    conditionType = cms.untracked.string('ESTimeSampleWeights'),
    since = cms.untracked.uint32(1),
    inputFile = cms.untracked.string('CondTools/Ecal/test/preshower/ESTimeSampleWeights_HG.txt')
    
    )))
    
process.p = cms.Path(process.ecalModule)
    
    

