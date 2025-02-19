import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
from EventFilter.CSCRawToDigi.cscUnpacker_cfi import *
from DQM.L1TMonitor.L1TCSCTPG_cfi import *
cscConditions = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    siteLocalConfig = cms.untracked.bool(False),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBGainsRcd'),
        tag = cms.string('CSCDBGains_ideal')
    ), 
        cms.PSet(
            record = cms.string('CSCDBCrosstalkRcd'),
            tag = cms.string('CSCDBCrosstalk_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCDBNoiseMatrixRcd'),
            tag = cms.string('CSCDBNoiseMatrix_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCDBPedestalsRcd'),
            tag = cms.string('CSCDBPedestals_ideal')
        ), 
        cms.PSet(
            record = cms.string('CSCChamberIndexRcd'),
            tag = cms.string('CSCChamberIndex')
        ), 
        cms.PSet(
            record = cms.string('CSCChamberMapRcd'),
            tag = cms.string('CSCChamberMap')
        ), 
        cms.PSet(
            record = cms.string('CSCCrateMapRcd'),
            tag = cms.string('CSCCrateMap')
        ), 
        cms.PSet(
            record = cms.string('CSCDDUMapRcd'),
            tag = cms.string('CSCDDUMap')
        )),
    messagelevel = cms.untracked.uint32(0),
    timetype = cms.string('runnumber'),
    #                string connect = "frontier://(serverurl=http://frontier1.cms:8000/FrontierOn)(serverurl=http://frontier2.cms:8000/FrontierOn)(retrieve-ziplevel=0)/CMS_COND_ON_18x_CSC"
    connect = cms.string('frontier://Frontier/CMS_COND_ON_18x_CSC'),
    authenticationMethod = cms.untracked.uint32(1)
)

l1tcsctpgpath = cms.Path(muonCSCDigis*l1tcsctpg)
muonCSCDigis.UnpackStatusDigis = True
#muonCSCDigis.isMTCCData = False
muonCSCDigis.ErrorMask = 0x0
muonCSCDigis.ExaminerMask = 0x16CBF3F6
muonCSCDigis.UseExaminer = True

