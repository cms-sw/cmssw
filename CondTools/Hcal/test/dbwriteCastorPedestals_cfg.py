import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")

#process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('./authentication.xml')
#process.CondDBCommon.connect = cms.string('sqlite_file:CastorPedestals.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')
#process.CondDBCommon.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_31X_HCAL')
process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_HCAL')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("CastorTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('Pedestals'),
        file = cms.FileInPath('CondTools/Hcal/test/castor_pedestals_run119814.txt')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
#    logconnect= cms.untracked.string('sqlite_file:log.db'),
    logconnect= cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),
#    logconnect= cms.untracked.string('oracle://cms_orcoff_prep/CMS_COND_31X_POPCONLOG'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CastorPedestalsRcd'),
        tag = cms.string('castor_pedestals_v1.0_hlt')
         ))
)

process.mytest = cms.EDAnalyzer("CastorPedestalsPopConAnalyzer",
    record = cms.string('CastorPedestalsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    #IOVRun=cms.untracked.uint32(119814)
    IOVRun=cms.untracked.uint32(164799)
    )
)

process.p = cms.Path(process.mytest)
