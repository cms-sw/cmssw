def initL1Subsystems( tagBaseVec = [],
                      objectKey = 'dummy' ):

    import FWCore.ParameterSet.Config as cms
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum

    if len( tagBaseVec ) == 0:
        from CondTools.L1Trigger.L1UniformTags_cfi import initL1UniformTags
        initL1UniformTags()
        tagBaseVec = initL1UniformTags.tagBaseVec

    initL1Subsystems.params = cms.PSet(
        recordInfo = cms.VPSet(
        cms.PSet(
            record = cms.string('L1JetEtScaleRcd'),
            tag = cms.string('L1JetEtScale_' + tagBaseVec[ L1CondEnum.L1JetEtScale ]),
            type = cms.string('L1CaloEtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1EmEtScaleRcd'),
            tag = cms.string('L1EmEtScale_' + tagBaseVec[ L1CondEnum.L1EmEtScale ]),
            type = cms.string('L1CaloEtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1HtMissScaleRcd'),
            tag = cms.string('L1HtMissScale_' + tagBaseVec[ L1CondEnum.L1HtMissScale ]),
            type = cms.string('L1CaloEtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1HfRingEtScaleRcd'),
            tag = cms.string('L1HfRingEtScale_' + tagBaseVec[ L1CondEnum.L1HfRingEtScale ]),
            type = cms.string('L1CaloEtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuTriggerScalesRcd'),
            tag = cms.string('L1MuTriggerScales_' + tagBaseVec[ L1CondEnum.L1MuTriggerScales ]),
            type = cms.string('L1MuTriggerScales'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuTriggerPtScaleRcd'),
            tag = cms.string('L1MuTriggerPtScale_' + tagBaseVec[ L1CondEnum.L1MuTriggerPtScale ]),
            type = cms.string('L1MuTriggerPtScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuGMTScalesRcd'),
            tag = cms.string('L1MuGMTScales_' + tagBaseVec[ L1CondEnum.L1MuGMTScales ]),
            type = cms.string('L1MuGMTScales'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCTFConfigurationRcd'),
            tag = cms.string('L1MuCSCTFConfiguration_' + tagBaseVec[ L1CondEnum.L1MuCSCTFConfiguration ]),
            type = cms.string('L1MuCSCTFConfiguration'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCTFAlignmentRcd'),
            tag = cms.string('L1MuCSCTFAlignment_' + tagBaseVec[ L1CondEnum.L1MuCSCTFAlignment ]),
            type = cms.string('L1MuCSCTFAlignment'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuCSCPtLutRcd'),
            tag = cms.string('L1MuCSCPtLut_' + tagBaseVec[ L1CondEnum.L1MuCSCPtLut ]),
            type = cms.string('L1MuCSCPtLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTEtaPatternLutRcd'),
            tag = cms.string('L1MuDTEtaPatternLut_' + tagBaseVec[ L1CondEnum.L1MuDTEtaPatternLut ]),
            type = cms.string('L1MuDTEtaPatternLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTExtLutRcd'),
            tag = cms.string('L1MuDTExtLut_' + tagBaseVec[ L1CondEnum.L1MuDTExtLut ]),
            type = cms.string('L1MuDTExtLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPhiLutRcd'),
            tag = cms.string('L1MuDTPhiLut_' + tagBaseVec[ L1CondEnum.L1MuDTPhiLut ]),
            type = cms.string('L1MuDTPhiLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTPtaLutRcd'),
            tag = cms.string('L1MuDTPtaLut_' + tagBaseVec[ L1CondEnum.L1MuDTPtaLut ]),
            type = cms.string('L1MuDTPtaLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTQualPatternLutRcd'),
            tag = cms.string('L1MuDTQualPatternLut_' + tagBaseVec[ L1CondEnum.L1MuDTQualPatternLut ]),
            type = cms.string('L1MuDTQualPatternLut'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1MuDTTFParametersRcd'),
            tag = cms.string('L1MuDTTFParameters_' + tagBaseVec[ L1CondEnum.L1MuDTTFParameters ]),
            type = cms.string('L1MuDTTFParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1RPCConfigRcd'),
            tag = cms.string('L1RPCConfig_' + tagBaseVec[ L1CondEnum.L1RPCConfig ]),
            type = cms.string('L1RPCConfig'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1RPCConeDefinitionRcd'),
            tag = cms.string('L1RPCConeDefinition_' + tagBaseVec[ L1CondEnum.L1RPCConeDefinition ]),
            type = cms.string('L1RPCConeDefinition'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1RPCHsbConfigRcd'),
            tag = cms.string('L1RPCHsbConfig_' + tagBaseVec[ L1CondEnum.L1RPCHsbConfig ]),
            type = cms.string('L1RPCHsbConfig'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1RPCBxOrConfigRcd'),
            tag = cms.string('L1RPCBxOrConfig_' + tagBaseVec[ L1CondEnum.L1RPCBxOrConfig ]),
            type = cms.string('L1RPCBxOrConfig'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1MuGMTParametersRcd'),
            tag = cms.string('L1MuGMTParameters_' + tagBaseVec[ L1CondEnum.L1MuGMTParameters ]),
            type = cms.string('L1MuGMTParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1RCTParametersRcd'),
            tag = cms.string('L1RCTParameters_' + tagBaseVec[ L1CondEnum.L1RCTParameters ]),
            type = cms.string('L1RCTParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1CaloEcalScaleRcd'),
            tag = cms.string('L1CaloEcalScale_' + tagBaseVec[ L1CondEnum.L1CaloEcalScale ]),
            type = cms.string('L1CaloEcalScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1CaloHcalScaleRcd'),
            tag = cms.string('L1CaloHcalScale_' + tagBaseVec[ L1CondEnum.L1CaloHcalScale ]),
            type = cms.string('L1CaloHcalScale'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GctJetFinderParamsRcd'),
            tag = cms.string('L1GctJetFinderParams_' + tagBaseVec[ L1CondEnum.L1GctJetFinderParams ]),
            type = cms.string('L1GctJetFinderParams'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtBoardMapsRcd'),
            tag = cms.string('L1GtBoardMaps_' + tagBaseVec[ L1CondEnum.L1GtBoardMaps ]),
            type = cms.string('L1GtBoardMaps'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtParametersRcd'),
            tag = cms.string('L1GtParameters_' + tagBaseVec[ L1CondEnum.L1GtParameters ]),
            type = cms.string('L1GtParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtStableParametersRcd'),
            tag = cms.string('L1GtStableParameters_' + tagBaseVec[ L1CondEnum.L1GtStableParameters ]),
            type = cms.string('L1GtStableParameters'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoAlgoTrig_' + tagBaseVec[ L1CondEnum.L1GtTriggerMaskVetoAlgoTrig ]),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtTriggerMenuRcd'),
            tag = cms.string('L1GtTriggerMenu_' + tagBaseVec[ L1CondEnum.L1GtTriggerMenu ]),
            type = cms.string('L1GtTriggerMenu'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1GtPsbSetupRcd'),
            tag = cms.string('L1GtPsbSetup_' + tagBaseVec[ L1CondEnum.L1GtPsbSetup ]),
            type = cms.string('L1GtPsbSetup'),
            key = cms.string(objectKey)
        ), 
        cms.PSet(
            record = cms.string('L1CaloGeometryRecord'),
            tag = cms.string('L1CaloGeometry_' + tagBaseVec[ L1CondEnum.L1CaloGeometry ]),
            type = cms.string('L1CaloGeometry'),
            key = cms.string(objectKey)
        ))
        )

    from CondTools.L1Trigger.L1RSSubsystemParams_cfi import initL1RSSubsystems
    initL1RSSubsystems( tagBaseVec,
                        objectKey,
                        objectKey,
                        objectKey,
                        objectKey,
                        objectKey,
                        objectKey,
                        objectKey,
                        objectKey,
                        objectKey )
    initL1Subsystems.params.recordInfo.extend(initL1RSSubsystems.params.recordInfo)
