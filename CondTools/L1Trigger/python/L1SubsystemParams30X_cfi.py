def initL1Subsystems( tagBase = 'IDEAL',
                      objectKey = 'dummy' ):

    import FWCore.ParameterSet.Config as cms

    initL1Subsystems.params = cms.PSet(
        recordInfo = cms.VPSet(
        cms.PSet(
            record = cms.string('L1JetEtScaleRcd'),
            tag = cms.string('L1JetEtScale_' + tagBase),
            type = cms.string('L1CaloEtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1EmEtScaleRcd'),
            tag = cms.string('L1EmEtScale_' + tagBase),
            type = cms.string('L1CaloEtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuTriggerScalesRcd'),
            tag = cms.string('L1MuTriggerScales_' + tagBase),
            type = cms.string('L1MuTriggerScales'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuTriggerPtScaleRcd'),
            tag = cms.string('L1MuTriggerPtScale_' + tagBase),
            type = cms.string('L1MuTriggerPtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTScalesRcd'),
            tag = cms.string('L1MuGMTScales_' + tagBase),
            type = cms.string('L1MuGMTScales'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCTFConfigurationRcd'),
            tag = cms.string('L1MuCSCTFConfiguration_' + tagBase),
            type = cms.string('L1MuCSCTFConfiguration'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCTFAlignmentRcd'),
            tag = cms.string('L1MuCSCTFAlignment_' + tagBase),
            type = cms.string('L1MuCSCTFAlignment'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCPtLutRcd'),
            tag = cms.string('L1MuCSCPtLut_' + tagBase),
            type = cms.string('L1MuCSCPtLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTEtaPatternLutRcd'),
            tag = cms.string('L1MuDTEtaPatternLut_' + tagBase),
            type = cms.string('L1MuDTEtaPatternLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTExtLutRcd'),
            tag = cms.string('L1MuDTExtLut_' + tagBase),
            type = cms.string('L1MuDTExtLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPhiLutRcd'),
            tag = cms.string('L1MuDTPhiLut_' + tagBase),
            type = cms.string('L1MuDTPhiLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPtaLutRcd'),
            tag = cms.string('L1MuDTPtaLut_' + tagBase),
            type = cms.string('L1MuDTPtaLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTQualPatternLutRcd'),
            tag = cms.string('L1MuDTQualPatternLut_' + tagBase),
            type = cms.string('L1MuDTQualPatternLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTTFParametersRcd'),
            tag = cms.string('L1MuDTTFParameters_' + tagBase),
            type = cms.string('L1MuDTTFParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1RPCConfigRcd'),
            tag = cms.string('L1RPCConfig_' + tagBase),
            type = cms.string('L1RPCConfig'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTParametersRcd'),
            tag = cms.string('L1MuGMTParameters_' + tagBase),
            type = cms.string('L1MuGMTParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTChannelMaskRcd'),
            tag = cms.string('L1MuGMTChannelMask_' + tagBase),
            type = cms.string('L1MuGMTChannelMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1RCTParametersRcd'),
            tag = cms.string('L1RCTParameters_' + tagBase),
            type = cms.string('L1RCTParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1RCTChannelMaskRcd'),
            tag = cms.string('L1RCTChannelMask_' + tagBase),
            type = cms.string('L1RCTChannelMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1CaloEcalScaleRcd'),
            tag = cms.string('L1CaloEcalScale_' + tagBase),
            type = cms.string('L1CaloEcalScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1CaloHcalScaleRcd'),
            tag = cms.string('L1CaloHcalScale_' + tagBase),
            type = cms.string('L1CaloHcalScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GctChannelMaskRcd'),
            tag = cms.string('L1GctChannelMask_' + tagBase),
            type = cms.string('L1GctChannelMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GctHfLutSetupRcd'),
            tag = cms.string('L1GctHfLutSetup_' + tagBase),
            type = cms.string('L1GctHfLutSetup'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GctJetFinderParamsRcd'),
            tag = cms.string('L1GctJetFinderParams_' + tagBase),
            type = cms.string('L1GctJetFinderParams'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtBoardMapsRcd'),
            tag = cms.string('L1GtBoardMaps_' + tagBase),
            type = cms.string('L1GtBoardMaps'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtParametersRcd'),
            tag = cms.string('L1GtParameters_' + tagBase),
            type = cms.string('L1GtParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsAlgoTrig_' + tagBase),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsTechTrig_' + tagBase),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtStableParametersRcd'),
            tag = cms.string('L1GtStableParameters_' + tagBase),
            type = cms.string('L1GtStableParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskAlgoTrig_' + tagBase),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskTechTrig_' + tagBase),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoAlgoTrig_' + tagBase),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoTechTrig_' + tagBase),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMenuRcd'),
            tag = cms.string('L1GtTriggerMenu_' + tagBase),
            type = cms.string('L1GtTriggerMenu'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtPsbSetupRcd'),
            tag = cms.string('L1GtPsbSetup_' + tagBase),
            type = cms.string('L1GtPsbSetup'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1CaloGeometryRecord'),
            tag = cms.string('L1CaloGeometry_' + tagBase),
            type = cms.string('L1CaloGeometry'),
            key = cms.string(objectKey)
        ))
        )
