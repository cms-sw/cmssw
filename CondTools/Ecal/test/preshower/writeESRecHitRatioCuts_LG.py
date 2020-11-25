import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_39X_PRESHOWER'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:ESRecHitRatioCuts_LG.db'

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
    # RH ratio cuts
    record = cms.string('ESRecHitRatioCutsRcd'),
    tag = cms.string('ESRecHitRatioCuts_LG')
    )))

process.ecalModule = cms.EDAnalyzer("StoreESCondition",
                                    gain = cms.uint32(1), 
                                    logfile = cms.string('./logfile.log'),
                                    toPut = cms.VPSet(cms.PSet(

    # RH ratio cuts
    conditionType = cms.untracked.string('ESRecHitRatioCuts'),
    since = cms.untracked.uint32(250000),  
    inputFile = cms.untracked.string('CondTools/Ecal/test/preshower/ESRecHitRatioCuts_LG.txt')
    
    )))
    
process.p = cms.Path(process.ecalModule)
    
    

