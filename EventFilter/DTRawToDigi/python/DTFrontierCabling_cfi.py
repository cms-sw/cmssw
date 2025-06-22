import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
DTCabling = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTReadOutMappingRcd'),
        tag = cms.string('DTROMap')
    )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_DT'),
)


