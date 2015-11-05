import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
                              treshold=cms.untracked.string("INFO")
                              )
)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string('sqlite_file:testExample.db')
process.CondDB.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.es_ascii = cms.ESSource("HcalOmdsCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('LutMetadata'),
        tag = cms.string('hcal-respcorr-test-v1'),
        version = cms.string('dummy-obsolete'),
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
      rec_hit_calibration, 
      lut_granularity, 
      output_lut_threshold 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
            MAX(theview.interval_of_validity_begin) as iov_begin, 
            theview.channel_map_id 
     from 
            cms_hcl_hcal_cond.v_hcal_lut_chan_data_v1 theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
     group by 
            theview.channel_map_id 
     order by 
            theview.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_lut_chan_data_v1 sp 
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
       rctlsb, 
       nominal_gain, 
       -1 
FROM ( 
     select 
            MIN(theview.record_id) as record_id, 
	    MAX(theview.interval_of_validity_begin) as iov_begin 
     from 
            cms_hcl_hcal_cond.v_hcal_lut_metadata_v1 theview 
     where 
            tag_name=:1 
     AND 
            theview.interval_of_validity_begin<=:2 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_lut_metadata_v1 sp 
on 
fp.record_id=sp.record_id 
        ''')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    logconnect= cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('HcalLutMetadataRcd'),
        tag = cms.string('hcal_resp_corrs_trivial_mc')
         ))
)

process.mytest = cms.EDAnalyzer("HcalLutMetadataPopConAnalyzer",
    record = cms.string('HcalLutMetadataRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
