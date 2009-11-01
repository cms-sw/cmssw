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
        object = cms.string('L1TriggerObjects'),
        tag = cms.string('hcal-l1trigger-test-v1'),
        version = cms.string('hcal-l1trigger-test-v1'),
        subversion = cms.int32(1),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44,LHWM_VERSION=22'),
        query = cms.string('''
        SELECT o.OBJECTNAME, o.SUBDET, o.IETA, o.IPHI, o.DEPTH, o.TYPE, o.SECTION, o.ISPOSITIVEETA, o.SECTOR, o.MODULE, o.CHANNEL,
               o.AVERAGE_PEDESTAL, o.RESPONSE_CORRECTED_GAIN, o.FLAG,
               m.TRIGGER_OBJECT_METADATA_VALUE as LUT_TAG_NAME, n.TRIGGER_OBJECT_METADATA_VALUE as ALGO_NAME
        FROM CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS o
        inner join CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS_MDA m
        on
           --o.tag_name=m.tag_name
        --AND
           --o.version=m.version
        --AND
           m.TRIGGER_OBJECT_METADATA_NAME='lut tag'
        inner join CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS_MDA n
        on
           --o.tag_name=n.tag_name
        --AND 
           --o.version=n.version
        --AND
           n.TRIGGER_OBJECT_METADATA_NAME='lut tag'
        WHERE
        o.TAG_NAME=:1
        and
        o.VERSION=:2
        ''')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    logconnect= cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('HcalL1TriggerObjectsRcd'),
        tag = cms.string('hcal_resp_corrs_trivial_mc')
         ))
)

process.mytest = cms.EDAnalyzer("HcalL1TriggerObjectsPopConAnalyzer",
    record = cms.string('HcalL1TriggerObjectsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
