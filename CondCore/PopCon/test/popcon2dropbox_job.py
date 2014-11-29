from popcon2dropbox_job_conf import md
from popcon2dropbox_job_conf import process

process.conf_o2o = cms.EDAnalyzer("DTKeyedConfigPopConAnalyzer",
    name = cms.untracked.string('DTCCBConfig'),
    Source = cms.PSet(
        DBParameters = cms.PSet(
        ),
        onlineDB = cms.string('oracle://cms_omds_lb/CMS_DT_ELEC_CONF'),
        minBrick = cms.untracked.int32(0),
        maxBrick = cms.untracked.int32(99999999),
        minRun = cms.untracked.int32(146960),
        maxRun = cms.untracked.int32(999999999),
        tag = cms.string('DTCCBConfig_V06_hlt'),
        container = cms.string('keyedConfBricks'),
        onlineAuthentication = cms.string( str(md.authPath()) ),
        onlineAuthSys = cms.untracked.int32( int(md.authSys()) )
    ),
    #targetDBConnectionString = cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_DT'),
    targetDBConnectionString = cms.untracked.string(str(md.destinationDatabase())),
    authenticationPath = cms.untracked.string( str(md.authPath()) ),
    authenticationSystem = cms.untracked.int32( int(md.authSys()) ),
    SinceAppendMode = cms.bool(True),
    record = cms.string('DTCCBConfigRcd'),
    loggingOn = cms.untracked.bool(True),
    debug = cms.bool(False)
)

process.p = cms.Path(process.conf_o2o)

