import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("ProcessOne")

options = VarParsing.VarParsing()
options.register( "password"
                , "myToto"
                , VarParsing.VarParsing.multiplicity.singleton
                , VarParsing.VarParsing.varType.string
                , "the password"
                  )
options.parseArguments()

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100000000),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(100000000),
    interval = cms.uint64(1)
)

process.load("CondCore.CondDB.CondDB_cfi")

#process.CondDB.connect = 'sqlite_file:EcalTPGFineGrainEBIdMap_v2_hlt.db'
process.CondDB.connect = 'oracle://cms_orcon_prod/CMS_CONDITIONS'
process.CondDB.DBParameters.authenticationPath = ''

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB, 
#    logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),
   logconnect = cms.untracked.string('sqlite_file:log.db'),   
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
        tag = cms.string('EcalTPGFineGrainEBIdMap_v2_hlt')
    ))
)

process.Test1 = cms.EDAnalyzer("ExTestEcalTPGFineGrainEBIdMapAnalyzer",
    record = cms.string('EcalTPGFineGrainEBIdMapRcd'),
    loggingOn= cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
     firstRun = cms.string('200000'),
     lastRun = cms.string('10000000'),
     OnlineDBSID = cms.string('cms_omds_lb'),
#     OnlineDBSID = cms.string('cms_orcon_adg'),  test on lxplus
     OnlineDBUser = cms.string('cms_ecal_r'),
     OnlineDBPassword = cms.string( options.password ),
     LocationSource = cms.string('P5'),
     Location = cms.string('P5_Co'),
     GenTag = cms.string('GLOBAL'),
     RunType = cms.string('PHYSICS')
    )                            
)

process.p = cms.Path(process.Test1)
