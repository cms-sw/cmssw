def initL1UniformTags( tagBase = "IDEAL" ):

    import FWCore.ParameterSet.Config as cms
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum

    initL1UniformTags.tagBaseVec = []
    for i in range( 0, L1CondEnum.NumL1Cond ):
        initL1UniformTags.tagBaseVec.append( tagBase )
#        print i, initL1UniformTags.tagBaseVec[ i ]
