import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.source.firstRun = cms.untracked.uint32(67838)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG')

# choose the source of the HCAL channel quality conditions
#process.es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
process.es_prefer_GlobalTag = cms.ESPrefer("HcalOmdsCalibrations","es_omds")

process.GlobalTag.globaltag = 'GR09_31X_V5H::All'

process.es_omds = cms.ESSource("HcalOmdsCalibrations",
    input = cms.VPSet(cms.PSet(
        object = cms.string('ChannelQuality'),
        tag = cms.string('hcal-quality-test-v1'),
        version = cms.string('hcal-quality-test-v1'),
        subversion = cms.int32(1),
        iov_begin = cms.int32(100000),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44'),
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
               and 
                      cq.VERSION=:2 
               AND 
                      cq.interval_of_validity_begin<:3 
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


process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.generateLuts = cms.EDAnalyzer("HcalLutGenerator",
                                      tag = cms.string('CRAFTPhysicsV2'),
                                      HO_master_file = cms.string('inputLUTcoder_CRUZET_part4_HO.dat')
                                      )


process.p = cms.Path(process.generateLuts)

