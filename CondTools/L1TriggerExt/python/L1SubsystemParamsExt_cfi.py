def initL1SubsystemsExt( tagBaseVec = [],
                      objectKey = 'dummy' ):

    import FWCore.ParameterSet.Config as cms
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt

    if len( tagBaseVec ) == 0:
        from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
        initL1UniformTagsExt()
        tagBaseVec = initL1UniformTagsExt.tagBaseVec

    initL1SubsystemsExt.params = cms.PSet(
        recordInfo = cms.VPSet(
        cms.PSet(
            record = cms.string('L1TUtmTriggerMenuO2ORcd'),
            tag = cms.string('L1TUtmTriggerMenu_' + tagBaseVec[ L1CondEnumExt.L1TUtmTriggerMenu ]),
            type = cms.string('L1TUtmTriggerMenu'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1TGlobalPrescalesVetosO2ORcd'),
            tag = cms.string('L1TGlobalPrescalesVetos_' + tagBaseVec[ L1CondEnumExt.L1TGlobalPrescalesVetos ]),
            type = cms.string('L1TGlobalPrescalesVetos'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1TMuonGlobalParamsO2ORcd'),
            tag = cms.string('L1TMuonGlobalParams_' + tagBaseVec[ L1CondEnumExt.L1TMuonGlobalParams ]),
            type = cms.string('L1TMuonGlobalParams'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1TMuonEndcapParamsO2ORcd'),
            tag = cms.string('L1TMuonEndCapParams_' + tagBaseVec[ L1CondEnumExt.L1TMuonEndcapParams ]),
            type = cms.string('L1TMuonEndCapParams'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1TMuonEndcapForestO2ORcd'),
            tag = cms.string('L1TMuonEndCapForest_' + tagBaseVec[ L1CondEnumExt.L1TMuonEndcapForest ]),
            type = cms.string('L1TMuonEndCapForest'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1TMuonOverlapParamsO2ORcd'),
            tag = cms.string('L1TMuonOverlapParams_' + tagBaseVec[ L1CondEnumExt.L1TMuonOverlapParams ]),
            type = cms.string('L1TMuonOverlapParams'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1TMuonBarrelParamsO2ORcd'),
            tag = cms.string('L1TMuonBarrelParams_' + tagBaseVec[ L1CondEnumExt.L1TMuonBarrelParams ]),
            type = cms.string('L1TMuonBarrelParams'),
            key = cms.string(objectKey)
        ),
        cms.PSet(
            record = cms.string('L1TCaloParamsO2ORcd'),
            tag = cms.string('L1TCaloParams_' + tagBaseVec[ L1CondEnumExt.L1TCaloParams ]),
            type = cms.string('L1TCaloParams'),
            key = cms.string(objectKey)
        )
        )
    )

    from CondTools.L1TriggerExt.L1RSSubsystemParamsExt_cfi import initL1RSSubsystemsExt
    initL1RSSubsystemsExt( tagBaseVec )
    initL1SubsystemsExt.params.recordInfo.extend(initL1RSSubsystemsExt.params.recordInfo)
