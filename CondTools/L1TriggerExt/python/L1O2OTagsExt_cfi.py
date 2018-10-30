def initL1O2OTagsExt():

    import FWCore.ParameterSet.Config as cms
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt

    initL1O2OTagsExt.tagBaseVec = [None] * L1CondEnumExt.NumL1Cond

    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ] = "Stage2v0_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TriggerKeyExt ] = "Stage2v0_hlt"

    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TUtmTriggerMenu ] = "Stage2v0_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TGlobalPrescalesVetos ] = "Stage2v0_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TMuonBarrelParams ] = "Stage2v1_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TMuonOverlapParams ] = "Stage2v0_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TMuonEndCapParams ] = "Stage2v3_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TMuonEndCapForest ] = "Stage2v1_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TMuonGlobalParams ] = "Stage2v0_hlt"
    initL1O2OTagsExt.tagBaseVec[ L1CondEnumExt.L1TCaloParams ] = "Stage2v3_hlt"
    
#    for i in range( 0, L1CondEnumExt.NumL1Cond ):
#        print i, initL1O2OTagsExt.tagBaseVec[ i ]
