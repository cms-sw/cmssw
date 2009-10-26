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
        object = cms.string('RespCorrs'),
        tag = cms.string('hcal-respcorr-test-v1'),
        version = cms.string('hcal-respcorr-test-v1'),
        subversion = cms.int32(1),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44,LHWM_VERSION=22'),
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
      VALUE 
FROM ( 
     select 
            MIN(rc.record_id) as record_id, 
	    MAX(rc.interval_of_validity_begin) as iov_begin, 
	    rc.channel_map_id 
     from 
            cms_hcl_hcal_cond.v_hcal_response_corrections rc 
     where 
            tag_name=:1 
     AND 
            rc.interval_of_validity_begin<=:2 
     group by 
            rc.channel_map_id 
     order by 
            rc.channel_map_id 
) fp 
inner join CMS_HCL_HCAL_COND.V_HCAL_RESPONSE_CORRECTIONS sp 
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
        record = cms.string('HcalRespCorrsRcd'),
        tag = cms.string('hcal_resp_corrs_trivial_mc')
         ))
)

process.mytest = cms.EDAnalyzer("HcalRespCorrsPopConAnalyzer",
    record = cms.string('HcalRespCorrsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
