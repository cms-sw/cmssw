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
        object = cms.string('ElectronicsMap'),
        tag = cms.string('DUMMY-TAG-TEST001'),
        version = cms.string('TEST:1'),
        subversion = cms.int32(1),
        accessor = cms.string('occi://CMS_HCL_APPUSER_R@anyhost/cms_omds_lb?PASSWORD=HCAL_Reader_44,LHWM_VERSION=22'),
        query = cms.string('''
        SELECT OBJECTNAME, SUBDET, IETA, IPHI, DEPTH, TYPE, SECTION, ISPOSITIVEETA, SECTOR, MODULE, CHANNEL, 
               1107314060 as i, 0 as cr, 2 as sl, 't' as tb, 2 as dcc, 0 as spigot, 1 as fiber, 0 as fiberchan 
        FROM CMS_HCL_HCAL_COND.V_HCAL_L1_TRIGGER_OBJECTS 
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
        record = cms.string('HcalElectronicsMapRcd'),
        tag = cms.string('hcal_emap_trivial_mc')
         ))
)

process.mytest = cms.EDAnalyzer("HcalElectronicsMapPopConAnalyzer",
    record = cms.string('HcalElectronicsMapRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
#    firstSince=cms.untracked.double(300) 
    IOVRun=cms.untracked.uint32(1)
    )
)

process.p = cms.Path(process.mytest)
