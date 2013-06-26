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
        version = cms.string('obsolete'),
        subversion = cms.int32(1),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44'),
        query = cms.string('''
SELECT 
      OBJECTNAME, 
      SUBDET, 
      IETA, 
      IPHI, 
      DEPTH, 
      TYPE, 
      SECTION, 
      ISPOSITIVEETA, 
      SECTOR, 
      MODULE, 
      CHANNEL, 
      AVERAGE_PEDESTAL, 
      RESPONSE_CORRECTED_GAIN, 
      FLAG, 
      'fake_metadata_name', 
      'fake_metadata_value' 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
            MAX(theview.interval_of_validity_begin) as iov_begin, 
            theview.channel_map_id 
     from 
            cms_hcl_hcal_cond.v_hcal_L1_TRIGGER_OBJECTS theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
     group by 
            theview.channel_map_id 
     order by 
            theview.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS sp 
on 
fp.record_id=sp.record_id 
 
union 
 
SELECT 
       'fakeobjectname', 
       'fakesubdetector', 
       -1, 
       -1, 
       -1, 
       -1, 
       'fakesection', 
       -1, 
       -1, 
       -1, 
       -1, 
       -999999.0, 
       -999999.0, 
       -999999, 
       TRIGGER_OBJECT_METADATA_NAME,
       TRIGGER_OBJECT_METADATA_VALUE 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
	    MAX(theview.interval_of_validity_begin) as iov_begin 
     from 
            cms_hcl_hcal_cond.v_hcal_L1_TRIGGER_OBJECTS_MDA theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS_MDA sp 
on 
fp.record_id=sp.record_id 
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
