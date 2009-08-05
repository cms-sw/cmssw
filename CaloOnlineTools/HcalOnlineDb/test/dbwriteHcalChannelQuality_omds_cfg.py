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

process.es_omds = cms.ESSource("HcalOmdsCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ChannelQuality'),
        tag = cms.string('hcal-quality-test-v1'),
        version = cms.string('hcal-quality-test-v1'),
        subversion = cms.int32(1),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44,LHWM_VERSION=22'),
        query = cms.string('''
        SELECT OBJECTNAME, SUBDET, IETA, IPHI, DEPTH, TYPE, SECTION, ISPOSITIVEETA, SECTOR, MODULE, CHANNEL,
               CHANNEL_STATUS_WORD 
        FROM CMS_HCL_HCAL_COND.V_HCAL_CHANNEL_QUALITY
        --SELECT zero_suppression, z*eta as ieta, phi, depth, detector_name as subdetector 
          --FROM CMS_HCL_HCAL_COND.V_HCAL_ZERO_SUPPRESSION 
        WHERE
          TAG_NAME=:1
        and
          VERSION=:2
        ''')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    logconnect= cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('HcalChannelQualityRcd'),
        tag = cms.string('hcal_channelStatus_trivial_mc')
         ))
)

process.mytest = cms.EDAnalyzer("HcalChannelQualityPopConAnalyzer",
    record = cms.string('HcalChannelQualityRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
