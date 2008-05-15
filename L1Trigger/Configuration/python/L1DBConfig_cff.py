import FWCore.ParameterSet.Config as cms

from L1Trigger.Configuration.L1DummyConfig_cff import *
from CondCore.DBCommon.CondDBCommon_cfi import *
l1pooldb = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1JetEtScaleRcd'),
        tag = cms.string('L1JetEtScale_CRUZET_hlt'),
        type = cms.string('L1CaloEtScale')
    ), 
        cms.PSet(
            record = cms.string('L1EmEtScaleRcd'),
            tag = cms.string('L1EmEtScale_CRUZET_hlt'),
            type = cms.string('L1CaloEtScale')
        ), 
        cms.PSet(
            record = cms.string('L1CSCTPParametersRcd'),
            tag = cms.string('L1CSCTPParameters_CRUZET_hlt'),
            type = cms.string('L1CSCTPParameters')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTEtaPatternLutRcd'),
            tag = cms.string('L1MuDTEtaPatternLut_CRUZET_hlt'),
            type = cms.string('L1MuDTEtaPatternLut')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTExtLutRcd'),
            tag = cms.string('L1MuDTExtLut_CRUZET_hlt'),
            type = cms.string('L1MuDTExtLut')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPhiLutRcd'),
            tag = cms.string('L1MuDTPhiLut_CRUZET_hlt'),
            type = cms.string('L1MuDTPhiLut')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPtaLutRcd'),
            tag = cms.string('L1MuDTPtaLut_CRUZET_hlt'),
            type = cms.string('L1MuDTPtaLut')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTQualPatternLutRcd'),
            tag = cms.string('L1MuDTQualPatternLut_CRUZET_hlt'),
            type = cms.string('L1MuDTQualPatternLut')
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTParametersRcd'),
            tag = cms.string('L1MuGMTParameters_CRUZET_hlt'),
            type = cms.string('L1MuGMTParameters')
        ), 
        cms.PSet(
            record = cms.string('L1RCTParametersRcd'),
            tag = cms.string('L1RCTParameters_CRUZET_hlt'),
            type = cms.string('L1RCTParameters')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetFinderParamsRcd'),
            tag = cms.string('L1GctJetFinderParams_CRUZET_hlt'),
            type = cms.string('L1GctJetFinderParams')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCalibFunRcd'),
            tag = cms.string('L1GctJetEtCalibrationFunction_CRUZET_hlt'),
            type = cms.string('L1GctJetEtCalibrationFunction')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCounterNegativeEtaRcd'),
            tag = cms.string('L1GctJetCounterNegativeEta_CRUZET_hlt'),
            type = cms.string('L1GctJetCounterSetup')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCounterPositiveEtaRcd'),
            tag = cms.string('L1GctJetCounterPositiveEta_CRUZET_hlt'),
            type = cms.string('L1GctJetCounterSetup')
        ), 
        cms.PSet(
            record = cms.string('L1GtBoardMapsRcd'),
            tag = cms.string('L1GtBoardMaps_CRUZET_hlt'),
            type = cms.string('L1GtBoardMaps')
        ), 
        cms.PSet(
            record = cms.string('L1GtParametersRcd'),
            tag = cms.string('L1GtParameters_CRUZET_hlt'),
            type = cms.string('L1GtParameters')
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsRcd'),
            tag = cms.string('L1GtPrescaleFactors_CRUZET_hlt'),
            type = cms.string('L1GtPrescaleFactors')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCounterNegativeEtaRcd'),
            tag = cms.string('L1GctJetCounterNegativeEta_CRUZET_hlt'),
            type = cms.string('L1GctJetCounterSetup')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCounterPositiveEtaRcd'),
            tag = cms.string('L1GctJetCounterPositiveEta_CRUZET_hlt'),
            type = cms.string('L1GctJetCounterSetup')
        ), 
        cms.PSet(
            record = cms.string('L1GtBoardMapsRcd'),
            tag = cms.string('L1GtBoardMaps_CRUZET_hlt'),
            type = cms.string('L1GtBoardMaps')
        ), 
        cms.PSet(
            record = cms.string('L1GtParametersRcd'),
            tag = cms.string('L1GtParameters_CRUZET_hlt'),
            type = cms.string('L1GtParameters')
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsRcd'),
            tag = cms.string('L1GtPrescaleFactors_CRUZET_hlt'),
            type = cms.string('L1GtPrescaleFactors')
        ), 
        cms.PSet(
            record = cms.string('L1GtStableParametersRcd'),
            tag = cms.string('L1GtStableParameters_CRUZET_hlt'),
            type = cms.string('L1GtStableParameters')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskRcd'),
            tag = cms.string('L1GtTriggerMask_CRUZET_hlt'),
            type = cms.string('L1GtTriggerMask')
        ))
)

es_prefer_l1pooldb = cms.ESPrefer("PoolDBESSource","l1pooldb")
l1pooldb.connect = 'frontier://cmsfrontier.cern.ch:8000/FrontierProd/CMS_COND_20X_L1T'

