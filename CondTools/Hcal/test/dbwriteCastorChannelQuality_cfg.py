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
process.CondDBCommon.connect = cms.string('sqlite_file:testExample.db')
#process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/nfshome0/popcondev/conddb')
#process.CondDBCommon.connect = cms.string('oracle://cms_orcon_prod/CMS_COND_31X_HCAL')
#process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('./authentication.xml')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("CastorTextCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ChannelQuality'),
        file = cms.FileInPath('CondFormats/CastorObjects/data/castor_quality.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run124009-132477.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run132478-132661.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run132662-132957.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run132958-138559.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run138560-138750.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run138751-140330.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run140331-140400.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run140401-141955.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run141956-148858.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run148859-148951.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run148952-150430.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run150431-150589.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run150590-158416.txt')
        #file = cms.FileInPath('CondFormats/CastorObjects/data/castor_channelquality_Run158417.txt')

    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    #logconnect= cms.untracked.string('oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'),
    logconnect= cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CastorChannelQualityRcd'),
        tag = cms.string('CastorChannelQuality_v2.2_offline')
         ))
)

process.mytest = cms.EDAnalyzer("CastorChannelQualityPopConAnalyzer",
    record = cms.string('CastorChannelQualityRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    #IOVRun=cms.untracked.uint32(124009)
    #IOVRun=cms.untracked.uint32(132478)
    #IOVRun=cms.untracked.uint32(132662)
    #IOVRun=cms.untracked.uint32(132958)
    #IOVRun=cms.untracked.uint32(138560)
    #IOVRun=cms.untracked.uint32(138751)
    #IOVRun=cms.untracked.uint32(140331)
    #IOVRun=cms.untracked.uint32(140401)
    #IOVRun=cms.untracked.uint32(141956)
    #IOVRun=cms.untracked.uint32(148859)
    #IOVRun=cms.untracked.uint32(148952)
    #IOVRun=cms.untracked.uint32(150431)
    #IOVRun=cms.untracked.uint32(150590)
    #IOVRun=cms.untracked.uint32(158417)
    )
)

process.p = cms.Path(process.mytest)
