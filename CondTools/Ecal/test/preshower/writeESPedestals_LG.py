import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_39X_PRESHOWER'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:ESPedestals_LG.db'

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
    # Pedestals
    record = cms.string('ESPedestalsRcd'),
    tag = cms.string('ESPedestals_LG')
    
    )))

process.ecalModule = cms.EDAnalyzer("StoreESCondition",
                                    logfile = cms.string('./logfile.log'),
                                    gain = cms.uint32(1),
                                    toPut = cms.VPSet(cms.PSet(

    # Pedestals
    conditionType = cms.untracked.string('ESPedestals'),
    since = cms.untracked.uint32(1),
    inputFile = cms.untracked.string('CondTools/Ecal/test/preshower/ESPedestals_LG.txt')
    
    )))
    
process.p = cms.Path(process.ecalModule)
    
    

