# The following comments couldn't be translated into the new config version:

#FrontierDev/CMS_COND_CSC"
import FWCore.ParameterSet.Config as cms

from CalibMuon.Configuration.getCSCDBConditionsME42_frontier_cff import *
cscConditions.connect = 'frontier://FrontierDev/CMS_COND_CSC'
cscConditions.toGet = cms.VPSet(cms.PSet(
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
    ))

