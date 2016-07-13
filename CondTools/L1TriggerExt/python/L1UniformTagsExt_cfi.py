def initL1UniformTagsExt( tagBase = "IDEAL" ):

    import FWCore.ParameterSet.Config as cms
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt

    initL1UniformTagsExt.tagBaseVec = []
    for i in range( 0, L1CondEnumExt.NumL1Cond ):
        initL1UniformTagsExt.tagBaseVec.append( tagBase )
#        print i, initL1UniformTagsExt.tagBaseVec[ i ]
