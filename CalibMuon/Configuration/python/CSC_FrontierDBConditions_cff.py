# The following comments couldn't be translated into the new config version:

#FrontierProd/CMS_COND_20X_CSC"
#	{ string record = "CSCChamberMapRcd"
#          string tag = "CSCChamberMap"},
#	{ string record = "CSCCrateMapRcd"
#          string tag = "CSCCrateMap"},

import FWCore.ParameterSet.Config as cms

from CalibMuon.Configuration.getCSCDBConditions_frontier_cff import *
cscConditions.connect = 'frontier://FrontierProd/CMS_COND_20X_CSC'
cscConditions.toGet = cms.VPSet(cms.PSet(
    record = cms.string('CSCDBGainsRcd'),
    tag = cms.string('CSCDBGains_mc')
), 
    cms.PSet(
        record = cms.string('CSCDBNoiseMatrixRcd'),
        tag = cms.string('CSCDBNoiseMatrix_mc')
    ), 
    cms.PSet(
        record = cms.string('CSCDBCrosstalkRcd'),
        tag = cms.string('CSCDBCrosstalk_mc')
    ), 
    cms.PSet(
        record = cms.string('CSCDBPedestalsRcd'),
        tag = cms.string('CSCDBPedestals_mc')
    ), 
    cms.PSet(
        record = cms.string('CSCChamberIndexRcd'),
        tag = cms.string('CSCChamberIndex')
    ), 
    cms.PSet(
        record = cms.string('CSCDDUMapRcd'),
        tag = cms.string('CSCDDUMap')
    ))

