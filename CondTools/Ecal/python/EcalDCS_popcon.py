import FWCore.ParameterSet.Config as cms
from CondCore.Utilities.popcon2dropbox_job_conf import options, psetForRecord, setup_popcon
import CondTools.Ecal.db_credentials as auth

recordName = "EcalDCSTowerStatusRcd"
tagTimeType = "runnumber"

process = setup_popcon( recordName, tagTimeType )



process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout')
)

process.essource = cms.ESSource("PoolDBESSource",
                                connect = cms.string( str(options.destinationDatabase) ),
                                DumpStat=cms.untracked.bool(True),
                                toGet = cms.VPSet( psetForRecord( recordName ) )
)

db_reader_account = 'CMS_ECAL_R'
db_service,db_user,db_pwd = auth.get_db_credentials( db_reader_account )

process.conf_o2o = cms.EDAnalyzer("ExTestEcalDCSAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string(recordName),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
        firstRun = cms.string('210000'),
        lastRun = cms.string('100000000'),
        OnlineDBUser = cms.string(db_user),
        debug = cms.bool(True),
        OnlineDBPassword = cms.string(db_pwd),
        OnlineDBSID = cms.string(db_service)
    ),
    targetDBConnectionString = cms.untracked.string(str(options.destinationDatabase))
)

process.p = cms.Path(process.conf_o2o)

