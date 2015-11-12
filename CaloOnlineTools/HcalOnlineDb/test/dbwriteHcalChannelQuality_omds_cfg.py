import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
                              threshold=cms.untracked.string("INFO")
                              )
)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string('sqlite_file:testExample.db')
process.CondDB.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(2),
    interval = cms.uint64(1)
)

process.es_omds = cms.ESSource("HcalOmdsCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ChannelQuality'),
        #tag = cms.string('hcal-quality-test-v1'),
        tag = cms.string('gak_v2'),
        version = cms.string('dummy-obsolete'),
        subversion = cms.int32(1),
        iov_begin = cms.int32(100),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44,LHWM_VERSION=22'),
        query = cms.string('''
        select 
               sp.objectname 
               ,sp.subdet as subdetector 
               ,sp.ieta as IETA 
               ,sp.iphi as IPHI 
               ,sp.depth as DEPTH 
               ,sp.type 
               ,sp.section 
               ,sp.ispositiveeta 
               ,sp.sector 
               ,sp.module 
               ,sp.channel_on_off_state as ON_OFF 
               ,sp.channel_status_word as STATUS_WORD 
    from 
               ( 
               select 
                      MAX(cq.record_id) as record_id 
                      ,MAX(cq.interval_of_validity_begin) as iov_begin 
                      ,cq.channel_map_id 
               from 
                      cms_hcl_hcal_cond.v_hcal_channel_quality cq 
               where 
                      tag_name=:1 
               --and 
                      --cq.VERSION=:2 
               AND 
                      cq.interval_of_validity_begin<=:2 
               group by 
                      cq.channel_map_id 
               order by 
                      cq.channel_map_id 
               ) fp 
        inner join 
               cms_hcl_hcal_cond.v_hcal_channel_quality sp 
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
        record = cms.string('HcalChannelQualityRcd'),
        #tag = cms.string('hcal_channelStatus_v1.00_test')
        tag = cms.string('gak_v2')
         ))
)

process.mytest = cms.EDAnalyzer("HcalChannelQualityPopConAnalyzer",
    record = cms.string('HcalChannelQualityRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(100)
    )
)

process.p = cms.Path(process.mytest)
