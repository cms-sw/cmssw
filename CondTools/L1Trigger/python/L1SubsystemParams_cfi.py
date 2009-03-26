import FWCore.ParameterSet.Config as cms


L1SubsystemParams = cms.PSet(
    recordInfo = cms.VPSet(
        cms.PSet(
            record = cms.string('L1JetEtScaleRcd'),
            tag = cms.string('L1JetEtScale_IDEAL'),
            type = cms.string('L1CaloEtScale'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1EmEtScaleRcd'),
            tag = cms.string('L1EmEtScale_IDEAL'),
            type = cms.string('L1CaloEtScale'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuTriggerScalesRcd'),
            tag = cms.string('L1MuTriggerScales_IDEAL'),
            type = cms.string('L1MuTriggerScales'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuTriggerPtScaleRcd'),
            tag = cms.string('L1MuTriggerPtScale_IDEAL'),
            type = cms.string('L1MuTriggerPtScale'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTScalesRcd'),
            tag = cms.string('L1MuGMTScales_IDEAL'),
            type = cms.string('L1MuGMTScales'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCTFConfigurationRcd'),
            tag = cms.string('L1MuCSCTFConfiguration_IDEAL'),
            type = cms.string('L1MuCSCTFConfiguration'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCTFAlignmentRcd'),
            tag = cms.string('L1MuCSCTFAlignment_IDEAL'),
            type = cms.string('L1MuCSCTFAlignment'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCPtLutRcd'),
            tag = cms.string('L1MuCSCPtLut_IDEAL'),
            type = cms.string('L1MuCSCPtLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTEtaPatternLutRcd'),
            tag = cms.string('L1MuDTEtaPatternLut_IDEAL'),
            type = cms.string('L1MuDTEtaPatternLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTExtLutRcd'),
            tag = cms.string('L1MuDTExtLut_IDEAL'),
            type = cms.string('L1MuDTExtLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPhiLutRcd'),
            tag = cms.string('L1MuDTPhiLut_IDEAL'),
            type = cms.string('L1MuDTPhiLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPtaLutRcd'),
            tag = cms.string('L1MuDTPtaLut_IDEAL'),
            type = cms.string('L1MuDTPtaLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTQualPatternLutRcd'),
            tag = cms.string('L1MuDTQualPatternLut_IDEAL'),
            type = cms.string('L1MuDTQualPatternLut'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuDTTFParametersRcd'),
            tag = cms.string('L1MuDTTFParameters_IDEAL'),
            type = cms.string('L1MuDTTFParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1RPCConfigRcd'),
            tag = cms.string('L1RPCConfig_IDEAL'),
            type = cms.string('L1RPCConfig'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1RPCConeDefinitionRcd'),
            tag = cms.string('L1RPCConeDefinition_IDEAL'),
            type = cms.string('L1RPCConeDefinition'),
            key = cms.string('dummy')
        ),
        cms.PSet(
            record = cms.string('L1MuGMTParametersRcd'),
            tag = cms.string('L1MuGMTParameters_IDEAL'),
            type = cms.string('L1MuGMTParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTChannelMaskRcd'),
            tag = cms.string('L1MuGMTChannelMask_IDEAL'),
            type = cms.string('L1MuGMTChannelMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1RCTParametersRcd'),
            tag = cms.string('L1RCTParameters_IDEAL'),
            type = cms.string('L1RCTParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1RCTChannelMaskRcd'),
            tag = cms.string('L1RCTChannelMask_IDEAL'),
            type = cms.string('L1RCTChannelMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1CaloEcalScaleRcd'),
            tag = cms.string('L1CaloEcalScale_IDEAL'),
            type = cms.string('L1CaloEcalScale'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1CaloHcalScaleRcd'),
            tag = cms.string('L1CaloHcalScale_IDEAL'),
            type = cms.string('L1CaloHcalScale'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GctChannelMaskRcd'),
            tag = cms.string('L1GctChannelMask_IDEAL'),
            type = cms.string('L1GctChannelMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GctHfLutSetupRcd'),
            tag = cms.string('L1GctHfLutSetup_IDEAL'),
            type = cms.string('L1GctHfLutSetup'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GctJetFinderParamsRcd'),
            tag = cms.string('L1GctJetFinderParams_IDEAL'),
            type = cms.string('L1GctJetFinderParams'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtBoardMapsRcd'),
            tag = cms.string('L1GtBoardMaps_IDEAL'),
            type = cms.string('L1GtBoardMaps'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtParametersRcd'),
            tag = cms.string('L1GtParameters_IDEAL'),
            type = cms.string('L1GtParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsAlgoTrig_IDEAL'),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsTechTrig_IDEAL'),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtStableParametersRcd'),
            tag = cms.string('L1GtStableParameters_IDEAL'),
            type = cms.string('L1GtStableParameters'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskAlgoTrig_IDEAL'),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskTechTrig_IDEAL'),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoAlgoTrig_IDEAL'),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoTechTrig_IDEAL'),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMenuRcd'),
            tag = cms.string('L1GtTriggerMenu_IDEAL'),
            type = cms.string('L1GtTriggerMenu'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1GtPsbSetupRcd'),
            tag = cms.string('L1GtPsbSetup_IDEAL'),
            type = cms.string('L1GtPsbSetup'),
            key = cms.string('dummy')
        ), 
        cms.PSet(
            record = cms.string('L1CaloGeometryRecord'),
            tag = cms.string('L1CaloGeometry_IDEAL'),
            type = cms.string('L1CaloGeometry'),
            key = cms.string('dummy')
        ))
)

