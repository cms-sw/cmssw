import FWCore.ParameterSet.Config as cms

#used for real DB conditions
from CondCore.DBCommon.CondDBSetup_cfi import *
cscConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBGainsRcd'),
        tag = cms.string('CSCDBGains_ideal')
    ), 
        cms.PSet(
            record = cms.string('CSCDBNoiseMatrixRcd'),
            tag = cms.string('CSCDBNoiseMatrix_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCDBCrosstalkRcd'),
            tag = cms.string('CSCDBCrosstalk_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCDBPedestalsRcd'),
            tag = cms.string('CSCDBPedestals_ideal')
        )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_ALIGNMENT'), ##FrontierDev/CMS_COND_ALIGNMENT"

    siteLocalConfig = cms.untracked.bool(True)
)


