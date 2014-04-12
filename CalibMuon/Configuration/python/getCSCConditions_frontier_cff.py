import FWCore.ParameterSet.Config as cms

#used for real DB conditions
from CondCore.DBCommon.CondDBSetup_cfi import *
cscConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCGainsRcd'),
        tag = cms.string('CSCGains_ideal')
    ), 
        cms.PSet(
            record = cms.string('CSCNoiseMatrixRcd'),
            tag = cms.string('CSCNoiseMatrix_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCcrosstalkRcd'),
            tag = cms.string('CSCCrosstalk_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCPedestalsRcd'),
            tag = cms.string('CSCPedestals_ideal')
        )),
    connect = cms.string('frontier://FrontierDev/CMS_COND_CSC'), ##FrontierDev/CMS_COND_CSC"

    siteLocalConfig = cms.untracked.bool(True)
)


