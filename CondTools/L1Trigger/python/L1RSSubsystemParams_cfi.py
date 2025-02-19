def initL1RSSubsystems( tagBaseVec = [],
                        L1MuDTTFMasksRcdKey = 'dummy',
                        L1MuGMTChannelMaskRcdKey = 'dummy',
                        L1RCTChannelMaskRcdKey = 'dummy',
                        L1RCTNoisyChannelMaskRcdKey = 'dummy',
                        L1GctChannelMaskRcdKey = 'dummy',
                        L1GtPrescaleFactorsAlgoTrigRcdKey = 'dummy',
                        L1GtPrescaleFactorsTechTrigRcdKey = 'dummy',
                        L1GtTriggerMaskAlgoTrigRcdKey = 'dummy',
                        L1GtTriggerMaskTechTrigRcdKey = 'dummy',
                        L1GtTriggerMaskVetoTechTrigRcdKey = 'dummy',
                        includeL1RCTNoisyChannelMask = True):

    import FWCore.ParameterSet.Config as cms
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum

    if includeL1RCTNoisyChannelMask == True:
        initL1RSSubsystems.params = cms.PSet(
            recordInfo = cms.VPSet(
            cms.PSet(
            record = cms.string('L1MuDTTFMasksRcd'),
            tag = cms.string('L1MuDTTFMasks_' + tagBaseVec[ L1CondEnum.L1MuDTTFMasks ]),
            type = cms.string('L1MuDTTFMasks'),
            key = cms.string(L1MuDTTFMasksRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1MuGMTChannelMaskRcd'),
            tag = cms.string('L1MuGMTChannelMask_' + tagBaseVec[ L1CondEnum.L1MuGMTChannelMask ]),
            type = cms.string('L1MuGMTChannelMask'),
            key = cms.string(L1MuGMTChannelMaskRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1RCTChannelMaskRcd'),
            tag = cms.string('L1RCTChannelMask_' + tagBaseVec[ L1CondEnum.L1RCTChannelMask ]),
            type = cms.string('L1RCTChannelMask'),
            key = cms.string(L1RCTChannelMaskRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1RCTNoisyChannelMaskRcd'),
            tag = cms.string('L1RCTNoisyChannelMask_' + tagBaseVec[ L1CondEnum.L1RCTNoisyChannelMask ]),
            type = cms.string('L1RCTNoisyChannelMask'),
            key = cms.string(L1RCTNoisyChannelMaskRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GctChannelMaskRcd'),
            tag = cms.string('L1GctChannelMask_' + tagBaseVec[ L1CondEnum.L1GctChannelMask ]),
            type = cms.string('L1GctChannelMask'),
            key = cms.string(L1GctChannelMaskRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsAlgoTrig_' + tagBaseVec[ L1CondEnum.L1GtPrescaleFactorsAlgoTrig ]),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(L1GtPrescaleFactorsAlgoTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsTechTrig_' + tagBaseVec[ L1CondEnum.L1GtPrescaleFactorsTechTrig ]),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(L1GtPrescaleFactorsTechTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskAlgoTrig_' + tagBaseVec[ L1CondEnum.L1GtTriggerMaskAlgoTrig ]),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(L1GtTriggerMaskAlgoTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtTriggerMaskTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskTechTrig_' + tagBaseVec[ L1CondEnum.L1GtTriggerMaskTechTrig ]),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(L1GtTriggerMaskTechTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoTechTrig_' + tagBaseVec[ L1CondEnum.L1GtTriggerMaskVetoTechTrig ]),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(L1GtTriggerMaskVetoTechTrigRcdKey)
            ))
            )
    else:
        initL1RSSubsystems.params = cms.PSet(
            recordInfo = cms.VPSet(
            cms.PSet(
            record = cms.string('L1MuDTTFMasksRcd'),
            tag = cms.string('L1MuDTTFMasks_' + tagBaseVec[ L1CondEnum.L1MuDTTFMasks ]),
            type = cms.string('L1MuDTTFMasks'),
            key = cms.string(L1MuDTTFMasksRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1MuGMTChannelMaskRcd'),
            tag = cms.string('L1MuGMTChannelMask_' + tagBaseVec[ L1CondEnum.L1MuGMTChannelMask ]),
            type = cms.string('L1MuGMTChannelMask'),
            key = cms.string(L1MuGMTChannelMaskRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1RCTChannelMaskRcd'),
            tag = cms.string('L1RCTChannelMask_' + tagBaseVec[ L1CondEnum.L1RCTChannelMask ]),
            type = cms.string('L1RCTChannelMask'),
            key = cms.string(L1RCTChannelMaskRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GctChannelMaskRcd'),
            tag = cms.string('L1GctChannelMask_' + tagBaseVec[ L1CondEnum.L1GctChannelMask ]),
            type = cms.string('L1GctChannelMask'),
            key = cms.string(L1GctChannelMaskRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsAlgoTrig_' + tagBaseVec[ L1CondEnum.L1GtPrescaleFactorsAlgoTrig ]),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(L1GtPrescaleFactorsAlgoTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
            tag = cms.string('L1GtPrescaleFactorsTechTrig_' + tagBaseVec[ L1CondEnum.L1GtPrescaleFactorsTechTrig ]),
            type = cms.string('L1GtPrescaleFactors'),
            key = cms.string(L1GtPrescaleFactorsTechTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
            tag = cms.string('L1GtTriggerMaskAlgoTrig_' + tagBaseVec[ L1CondEnum.L1GtTriggerMaskAlgoTrig ]),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(L1GtTriggerMaskAlgoTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtTriggerMaskTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskTechTrig_' + tagBaseVec[ L1CondEnum.L1GtTriggerMaskTechTrig ]),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(L1GtTriggerMaskTechTrigRcdKey)
            ), 
            cms.PSet(
            record = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
            tag = cms.string('L1GtTriggerMaskVetoTechTrig_' + tagBaseVec[ L1CondEnum.L1GtTriggerMaskVetoTechTrig ]),
            type = cms.string('L1GtTriggerMask'),
            key = cms.string(L1GtTriggerMaskVetoTechTrigRcdKey)
            ))
            )
