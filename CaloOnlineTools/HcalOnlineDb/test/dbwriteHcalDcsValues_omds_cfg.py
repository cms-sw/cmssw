import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
                              threshold=cms.untracked.string("INFO")
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
        object = cms.string('DcsValues'),
        tag = cms.string('HcalDcsValues_v1.00_offline'),
        version = cms.string('dummy-obsolete'),
        subversion = cms.int32(1),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44'),
        #accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/CMSDEVR_LB?PASSWORD=HCAL_Reader_44'),
        iov_begin = cms.int32(123596),
        query = cms.string('''
select 
      i.dpname, 
      -1 as lumisection, 
      i.value,          
      i.set_high as upper,   
      i.set_low as lower,       
      i.tag_name as tag, 
      'fakeversion' as version, 
      1 as subversion          
from                           
      cms_hcl_hcal_cond.V_CMS_HCAL_HV_INIT_VALUES i 
where                 
      i.tag_name like :1 
      and length(:2)>-1 and :3>-1 
      and i.run_number = :4      
UNION 
select 
      i.dpname, 
      -1 as lumisection, 
      i.value,          
      i.set_high as upper,   
      i.set_low as lower,       
      i.tag_name as tag, 
      'fakeversion' as version, 
      1 as subversion          
from                           
      cms_hcl_hcal_cond.V_CMS_HCAL_HV_UPDATE_VALUES i 
where                 
      i.tag_name like :1 
      and length(:2)>-1 and :3>-1 
      and i.run_number = :4 
      ''')
    ))
)


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    logconnect= cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('HcalDcsRcd'),
        tag = cms.string('HcalDcsValues_v1.00_offline')
         ))
)

process.mytest = cms.EDAnalyzer("HcalDcsValuesPopConAnalyzer",
    record = cms.string('HcalDcsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
