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
process.CondDBCommon.connect = cms.string('sqlite_file:testExample2.db')
#process.CondDBCommon.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_HCAL')
#process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("CastorTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('RecoParams'),
        file = cms.FileInPath('CondFormats/CastorObjects/data/castor_recoparams_Run1-129455.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_recoparams_Run129456.txt')
# for MC
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_recoparams_Run1_mc.txt')

    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    logconnect= cms.untracked.string('sqlite_file:log2.db'),
    #logconnect= cms.untracked.string('oracle://cms_orcoff_prep/CMS_COND_POPCONLOG'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CastorRecoParamsRcd'),
        tag = cms.string('CastorRecoParams_v1.00_offline')
         ))
)

process.mytest = cms.EDAnalyzer("CastorRecoParamsPopConAnalyzer",
    record = cms.string('CastorRecoParamsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    #IOVRun=cms.untracked.uint32(129456)
    )
)

process.p = cms.Path(process.mytest)
