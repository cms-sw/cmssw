def initL1RSSubsystemsExt( tagBaseVec = [],
#                        L1MuDTTFMasksRcdKey = 'dummy',
):

    import FWCore.ParameterSet.Config as cms
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt

    initL1RSSubsystemsExt.params = cms.PSet( recordInfo = cms.VPSet() )
