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
#process.CondDBCommon.connect = cms.string('sqlite_file:CastorEmap.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#process.CondDBCommon.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_31X_HCAL')
process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_HCAL')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("CastorTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ElectronicsMap'),
        file = cms.FileInPath('CondFormats/CastorObjects/data/emap_dcc_nominal_Run121872.txt')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
#    logconnect= cms.untracked.string('sqlite_file:log.db'),
    #logconnect= cms.untracked.string('oracle://cms_orcoff_prep/CMS_COND_31X_POPCONLOG'),
    logconnect= cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CastorElectronicsMapRcd'),
        tag = cms.string('CastorElectronicsMap_v2.01_mc')
         ))
)

process.mytest = cms.EDAnalyzer("CastorElectronicsMapPopConAnalyzer",
    record = cms.string('CastorElectronicsMapRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
