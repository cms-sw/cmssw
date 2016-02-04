import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
                              threshold=cms.untracked.string("INFO")
                              )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('CONNECT_STRING')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('POOL_AUTH_PATH')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_omds = cms.ESSource("HcalOmdsCalibrations",
    input = cms.VPSet(cms.PSet(
    object = cms.string('CONDITION_TYPE'),
    tag = cms.string('OMDS_CONDITION_TAG'),
    version = cms.string('fakeversion'),
    subversion = cms.int32(1),
    iov_begin = cms.int32(OMDS_IOV),
    accessor = cms.string('OMDS_ACCESSOR_STRING'),
    query = cms.string('''
    OMDS_QUERY
    ''')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    logconnect= cms.untracked.string('POOL_LOGCONNECT'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('POOL_RECORD'),
        tag = cms.string('POOL_OUTPUT_TAG')
         ))
)

process.mytest = cms.EDAnalyzer("HcalCONDITION_TYPEPopConAnalyzer",
    record = cms.string('POOL_RECORD'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(POOL_IOV)
    )
)

process.p = cms.Path(process.mytest)
