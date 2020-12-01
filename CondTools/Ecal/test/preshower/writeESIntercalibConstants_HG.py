import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDBCommon.connect = 'oracle://cms_orcon_prod/CMS_COND_39X_PRESHOWER'
#process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'
process.CondDB.connect = 'sqlite_file:ESIntercalibConstants_HG.db'

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
    # MIP constants
    record = cms.string('ESIntercalibConstantsRcd'),
    # ESIntercalibConstants_V01_offline is used up to May 2012
    tag = cms.string('ESIntercalibConstants_HG') 
    
    )))

process.ecalModule = cms.EDAnalyzer("StoreESCondition",
                                    gain = cms.uint32(2), # HG
                                    #gain = cms.uint32(1), # LG
                                    logfile = cms.string('./logfile.log'),
                                    toPut = cms.VPSet(cms.PSet(

    # MIP constants 
    conditionType = cms.untracked.string('ESIntercalibConstants'),
    #inputFile = cms.untracked.string('CondTools/Ecal/test/preshower/calibration_constant_LG_Run2012A.txt')
    # 2012/11/28 LG 2012 D for test
    since = cms.untracked.uint32(1),
    inputFile = cms.untracked.string('CondTools/Ecal/test/preshower/ESIntercalibConstants_HG.txt')
    )))
    
process.p = cms.Path(process.ecalModule)
    
    

