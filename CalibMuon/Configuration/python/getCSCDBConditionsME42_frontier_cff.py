import FWCore.ParameterSet.Config as cms

#used for real DB conditions
from CondCore.DBCommon.CondDBSetup_cfi import *
cscConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBGainsRcd'),
        tag = cms.string('CSCDBGains_test')
    ), 
        cms.PSet(
            record = cms.string('CSCDBNoiseMatrixRcd'),
            tag = cms.string('CSCDBNoiseMatrix_test')
        ), 
        cms.PSet(
            record = cms.string('CSCDBCrosstalkRcd'),
            tag = cms.string('CSCDBCrosstalk_test')
        ), 
        cms.PSet(
            record = cms.string('CSCDBPedestalsRcd'),
            tag = cms.string('CSCDBPedestals_test')
        )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_ALIGNMENT'), ##FrontierDev/CMS_COND_ALIGNMENT"

    siteLocalConfig = cms.untracked.bool(True)
)


