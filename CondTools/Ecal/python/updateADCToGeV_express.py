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
    lastValue = cms.uint64(100000000000),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(100000000000),
    interval = cms.uint64(1)
)

process.load("CondCore.CondDB.CondDB_cfi")

process.CondDB.DBParameters.authenticationPath = ''

#process.CondDB.connect = 'sqlite_file:EcalADCToGeVConstant_V1_express.db'
process.CondDB.connect = 'oracle://cms_orcon_prod/CMS_CONDITIONS'


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB, 
#    logconnect = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),
   logconnect = cms.untracked.string('sqlite_file:log.db'),   
        toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalADCToGeVConstantRcd'),
        tag = cms.string('EcalADCToGeVConstant_V1_express')
    ))
)

process.Test1 = cms.EDAnalyzer("ExTestEcalADCToGeVAnalyzer",
    record = cms.string('EcalADCToGeVConstantRcd'),
    loggingOn= cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
     FileLowField = cms.string('/data/O2O/Ecal/TPG/ADCToGeV_Boff.xml'),
     FileHighField = cms.string('/data/O2O/Ecal/TPG/ADCToGeV_Bon.xml'),
#     FileLowField = cms.string('/afs/cern.ch/work/d/depasse/cmssw/CMSSW_8_0_1/src/CondTools/Ecal/python/ADCToGeV_express_current_BOFF.xml'),
#     FileHighField = cms.string('/afs/cern.ch/work/d/depasse/cmssw/CMSSW_8_0_1/src/CondTools/Ecal/python/ADCToGeV_express_current_BON.xml'),
     firstRun = cms.string('207149'),
     lastRun = cms.string('10000000'),
     OnlineDBSID = cms.string('cms_omds_lb'),
#     OnlineDBSID = cms.string('cms_orcon_adg'),  test on lxplus
     OnlineDBUser = cms.string('cms_ecal_r'),
     OnlineDBPassword = cms.string( options.password ),
     LocationSource = cms.string('P5'),
     Location = cms.string('P5_Co'),
     GenTag = cms.string('GLOBAL'),
     RunType = cms.string('COSMICS')
    )                            
)

process.p = cms.Path(process.Test1)
