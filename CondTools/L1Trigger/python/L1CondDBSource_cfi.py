import FWCore.ParameterSet.Config as cms
from CondCore.DBCommon.CondDBSetup_cfi import *

l1conddb = cms.ESSource("PoolDBESSource",
                        CondDBSetup,
                        connect = cms.string('frontier://FrontierPrep/CMS_COND_L1T'),
                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('L1JetEtScaleRcd'),
    tag = cms.string('L1JetEtScale_CRAFT_hlt'),
    type = cms.string('L1CaloEtScale'),
    ),
                                          cms.PSet(
    record = cms.string('L1EmEtScaleRcd'),
    tag = cms.string('L1EmEtScale_CRAFT_hlt'),
    type = cms.string('L1CaloEtScale'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuTriggerScalesRcd'),
    tag = cms.string('L1MuTriggerScales_CRAFT_hlt'),
    type = cms.string('L1MuTriggerScales'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuTriggerPtScaleRcd'),
    tag = cms.string('L1MuTriggerPtScale_CRAFT_hlt'),
    type = cms.string('L1MuTriggerPtScale'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuGMTScalesRcd'),
    tag = cms.string('L1MuGMTScales_CRAFT_hlt'),
    type = cms.string('L1MuGMTScales'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuCSCTFConfigurationRcd'),
    tag = cms.string('L1MuCSCTFConfiguration_CRAFT_hlt'),
    type = cms.string('L1MuCSCTFConfiguration'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuCSCTFAlignmentRcd'),
    tag = cms.string('L1MuCSCTFAlignment_CRAFT_hlt'),
    type = cms.string('L1MuCSCTFAlignment'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuCSCPtLutRcd'),
    tag = cms.string('L1MuCSCPtLut_CRAFT_hlt'),
    type = cms.string('L1MuCSCPtLut'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuDTEtaPatternLutRcd'),
    tag = cms.string('L1MuDTEtaPatternLut_CRAFT_hlt'),
    type = cms.string('L1MuDTEtaPatternLut'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuDTExtLutRcd'),
    tag = cms.string('L1MuDTExtLut_CRAFT_hlt'),
    type = cms.string('L1MuDTExtLut'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuDTPhiLutRcd'),
    tag = cms.string('L1MuDTPhiLut_CRAFT_hlt'),
    type = cms.string('L1MuDTPhiLut'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuDTPtaLutRcd'),
    tag = cms.string('L1MuDTPtaLut_CRAFT_hlt'),
    type = cms.string('L1MuDTPtaLut'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuDTQualPatternLutRcd'),
    tag = cms.string('L1MuDTQualPatternLut_CRAFT_hlt'),
    type = cms.string('L1MuDTQualPatternLut'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuDTTFParametersRcd'),
    tag = cms.string('L1MuDTTFParameters_CRAFT_hlt'),
    type = cms.string('L1MuDTTFParameters'),
    ),
                                          cms.PSet(
    record = cms.string('L1RPCConfigRcd'),
    tag = cms.string('L1RPCConfig_CRAFT_hlt'),
    type = cms.string('L1RPCConfig'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuGMTParametersRcd'),
    tag = cms.string('L1MuGMTParameters_CRAFT_hlt'),
    type = cms.string('L1MuGMTParameters'),
    ),
                                          cms.PSet(
    record = cms.string('L1MuGMTChannelMaskRcd'),
    tag = cms.string('L1MuGMTChannelMask_CRAFT_hlt'),
    type = cms.string('L1MuGMTChannelMask'),
    ),
                                          cms.PSet(
    record = cms.string('L1RCTParametersRcd'),
    tag = cms.string('L1RCTParameters_CRAFT_hlt'),
    type = cms.string('L1RCTParameters'),
    ),
                                          cms.PSet(
    record = cms.string('L1RCTChannelMaskRcd'),
    tag = cms.string('L1RCTChannelMask_CRAFT_hlt'),
    type = cms.string('L1RCTChannelMask'),
    ),
                                          cms.PSet(
    record = cms.string('L1CaloEcalScaleRcd'),
    tag = cms.string('L1CaloEcalScale_CRAFT_hlt'),
    type = cms.string('L1CaloEcalScale'),
    ),
                                          cms.PSet(
    record = cms.string('L1CaloHcalScaleRcd'),
    tag = cms.string('L1CaloHcalScale_CRAFT_hlt'),
    type = cms.string('L1CaloHcalScale'),
    ),
                                          cms.PSet(
    record = cms.string('L1GctChannelMaskRcd'),
    tag = cms.string('L1GctChannelMask_CRAFT_hlt'),
    type = cms.string('L1GctChannelMask'),
    ),
                                          cms.PSet(
    record = cms.string('L1GctHfLutSetupRcd'),
    tag = cms.string('L1GctHfLutSetup_CRAFT_hlt'),
    type = cms.string('L1GctHfLutSetup'),
    ),
                                          cms.PSet(
    record = cms.string('L1GctJetFinderParamsRcd'),
    tag = cms.string('L1GctJetFinderParams_CRAFT_hlt'),
    type = cms.string('L1GctJetFinderParams'),
    ),
                                          cms.PSet(
    record = cms.string('L1GctJetCalibFunRcd'),
    tag = cms.string('L1GctJetEtCalibrationFunction_CRAFT_hlt'),
    type = cms.string('L1GctJetEtCalibrationFunction'),
    ),
                                          cms.PSet(
    record = cms.string('L1GctJetCounterNegativeEtaRcd'),
    tag = cms.string('L1GctJetCounterNegativeEta_CRAFT_hlt'),
    type = cms.string('L1GctJetCounterSetup'),
    ),
                                          cms.PSet(
    record = cms.string('L1GctJetCounterPositiveEtaRcd'),
    tag = cms.string('L1GctJetCounterPositiveEta_CRAFT_hlt'),
    type = cms.string('L1GctJetCounterSetup'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtBoardMapsRcd'),
    tag = cms.string('L1GtBoardMaps_CRAFT_hlt'),
    type = cms.string('L1GtBoardMaps'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtParametersRcd'),
    tag = cms.string('L1GtParameters_CRAFT_hlt'),
    type = cms.string('L1GtParameters'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
    tag = cms.string('L1GtPrescaleFactorsAlgoTrig_CRAFT_hlt'),
    type = cms.string('L1GtPrescaleFactors'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
    tag = cms.string('L1GtPrescaleFactorsTechTrig_CRAFT_hlt'),
    type = cms.string('L1GtPrescaleFactors'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtStableParametersRcd'),
    tag = cms.string('L1GtStableParameters_CRAFT_hlt'),
    type = cms.string('L1GtStableParameters'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
    tag = cms.string('L1GtTriggerMaskAlgoTrig_CRAFT_hlt'),
    type = cms.string('L1GtTriggerMask'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtTriggerMaskTechTrigRcd'),
    tag = cms.string('L1GtTriggerMaskTechTrig_CRAFT_hlt'),
    type = cms.string('L1GtTriggerMask'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
    tag = cms.string('L1GtTriggerMaskVetoAlgoTrig_CRAFT_hlt'),
    type = cms.string('L1GtTriggerMask'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
    tag = cms.string('L1GtTriggerMaskVetoTechTrig_CRAFT_hlt'),
    type = cms.string('L1GtTriggerMask'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtTriggerMenuRcd'),
    tag = cms.string('L1GtTriggerMenu_CRAFT_hlt'),
    type = cms.string('L1GtTriggerMenu'),
    ),
                                          cms.PSet(
    record = cms.string('L1GtPsbSetupRcd'),
    tag = cms.string('L1GtPsbSetup_CRAFT_hlt'),
    type = cms.string('L1GtPsbSetup'),
    ),
                                          cms.PSet(
    record = cms.string('L1CaloGeometryRecord'),
    tag = cms.string('L1CaloGeometry_CRAFT_hlt'),
    type = cms.string('L1CaloGeometry'),
    )),
                        BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
                        )
es_prefer_l1conddb = cms.ESPrefer("PoolDBESSource","l1conddb")
