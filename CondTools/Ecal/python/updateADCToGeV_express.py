import FWCore.ParameterSet.Config as cms
import CondTools.Ecal.conddb_init as conddb_init
import CondTools.Ecal.db_credentials as auth

process = cms.Process("ProcessOne")

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

process.CondDB.connect = conddb_init.options.destinationDatabase


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB, 
        toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalADCToGeVConstantRcd'),
        tag = cms.string(conddb_init.options.destinationTag)
    ))
)

db_reader_account = 'CMS_ECAL_R'
db_service,db_user,db_pwd = auth.get_db_credentials( db_reader_account )

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
     OnlineDBSID = cms.string(db_service),
     OnlineDBUser = cms.string(db_user),
     OnlineDBPassword = cms.string( db_pwd ),
     LocationSource = cms.string('P5'),
     Location = cms.string('P5_Co'),
     GenTag = cms.string('GLOBAL'),
     RunType = cms.string('COSMICS')
    )                            
)

process.p = cms.Path(process.Test1)
