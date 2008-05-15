import FWCore.ParameterSet.Config as cms

# keys are used only in L1TriggerKeyDummyProd
# tags are used in L1CondDBIOVWriterProd and must match PoolDBESSource
L1SubsystemParams = cms.PSet(
    recordInfo = cms.VPSet(cms.PSet(
        record = cms.string('L1JetEtScaleRcd'),
        tag = cms.string('L1JetEtScale_CRUZET_hlt'),
        type = cms.string('L1CaloEtScale'),
        key = cms.string('dummy')
    ), 
        cms.PSet(
            record = cms.string('L1EmEtScaleRcd'),
            tag = cms.string('L1EmEtScale_CRUZET_hlt'),
            type = cms.string('L1CaloEtScale'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1CSCTPParametersRcd'),
            tag = cms.string('L1CSCTPParameters_CRUZET_hlt'),
            type = cms.string('L1CSCTPParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCTFConfigurationRcd'),
            tag = cms.string('L1MuCSCTFConfiguration_CRUZET_hlt'),
            type = cms.string('L1MuCSCTFConfiguration'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('DTConfigManagerRcd'),
            tag = cms.string('DTConfigManager_CRUZET_hlt'),
            type = cms.string('DTConfigManager'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTEtaPatternLutRcd'),
            tag = cms.string('L1MuDTEtaPatternLut_CRUZET_hlt'),
            type = cms.string('L1MuDTEtaPatternLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTExtLutRcd'),
            tag = cms.string('L1MuDTExtLut_CRUZET_hlt'),
            type = cms.string('L1MuDTExtLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPhiLutRcd'),
            tag = cms.string('L1MuDTPhiLut_CRUZET_hlt'),
            type = cms.string('L1MuDTPhiLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPtaLutRcd'),
            tag = cms.string('L1MuDTPtaLut_CRUZET_hlt'),
            type = cms.string('L1MuDTPtaLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTQualPatternLutRcd'),
            tag = cms.string('L1MuDTQualPatternLut_CRUZET_hlt'),
            type = cms.string('L1MuDTQualPatternLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTTFParametersRcd'),
            tag = cms.string('L1MuDTTFParameters_CRUZET_hlt'),
            type = cms.string('L1MuDTTFParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTParametersRcd'),
            tag = cms.string('L1MuGMTParameters_CRUZET_hlt'),
            type = cms.string('L1MuGMTParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1RCTParametersRcd'),
            tag = cms.string('L1RCTParameters_CRUZET_hlt'),
            type = cms.string('L1RCTParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1RCTChannelMaskRcd'),
            tag = cms.string('L1RCTChannelMask_CRUZET_hlt'),
            type = cms.string('L1RCTChannelMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetFinderParamsRcd'),
            tag = cms.string('L1GctJetFinderParams_CRUZET_hlt'),
            type = cms.string('L1GctJetFinderParams'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCalibFunRcd'),
            tag = cms.string('L1GctJetEtCalibrationFunction_CRUZET_hlt'),
            type = cms.string('L1GctJetEtCalibrationFunction'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCounterNegativeEtaRcd'),
            tag = cms.string('L1GctJetCounterNegativeEta_CRUZET_hlt'),
            type = cms.string('L1GctJetCounterSetup'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetCounterPositiveEtaRcd'),
            tag = cms.string('L1GctJetCounterPositiveEta_CRUZET_hlt'),
            type = cms.string('L1GctJetCounterSetup'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtBoardMapsRcd'),
            tag = cms.string('L1GtBoardMaps_CRUZET_hlt'),
            type = cms.string('L1GtBoardMaps'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtParametersRcd'),
            tag = cms.string('L1GtParameters_CRUZET_hlt'),
            type = cms.string('L1GtParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsAlgoTrig_CRUZET_hlt'),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsTechTrig_CRUZET_hlt'),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtStableParametersRcd'),
            tag = cms.string('L1GtStableParameters_CRUZET_hlt'),
            type = cms.string('L1GtStableParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskAlgoTrig_CRUZET_hlt'),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskTechTrig_CRUZET_hlt'),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string('dummy')
        ))
)

