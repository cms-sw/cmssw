import FWCore.ParameterSet.Config as cms

SiStripDBCabling = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('CSA07_SiStripFedCabling')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_STRIP')
)


