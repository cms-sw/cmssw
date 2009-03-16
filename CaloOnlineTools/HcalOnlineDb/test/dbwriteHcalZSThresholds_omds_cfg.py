import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
                              treshold=cms.untracked.string("INFO")
                              )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('sqlite_file:testExample.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("HcalOmdsCalibrations",
    input = cms.VPSet(cms.PSet(
        #object = cms.string('ChannelQuality'),
        object = cms.string('ZSThresholds'),
        #file = cms.FileInPath('occi://CMS_HCL_PRTTYPE_HCAL_READER@anyhost/int2r?PASSWORD=HCAL_Reader_88,LHWM_VERSION=22')
        #file = cms.FileInPath('CondFormats/HcalObjects/data/hcal_channelStatus_default.txt')
        tag = cms.string('GREN_ZS_9adc_v2')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    logconnect= cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        #record = cms.string('HcalChannelQualityRcd'),
        #tag = cms.string('hcal_channelStatus_trivial_mc')
        record = cms.string('HcalZSThresholdsRcd'),
        tag = cms.string('hcal_zs_thresholds_trivial_mc')
         ))
)

#process.mytest = cms.EDAnalyzer("HcalChannelQualityPopConAnalyzer",
process.mytest = cms.EDAnalyzer("HcalZSThresholdsPopConAnalyzer",
    #record = cms.string('HcalChannelQualityRcd'),
    record = cms.string('HcalZSThresholdsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
