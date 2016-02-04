def initPayloadWriter( process,
                       outputDBConnect = 'sqlite_file:l1config.db',
                       outputDBAuth = '.',
                       tagBaseVec = [] ):
    import FWCore.ParameterSet.Config as cms
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum

    if len( tagBaseVec ) == 0:
        from CondTools.L1Trigger.L1UniformTags_cfi import initL1UniformTags
        initL1UniformTags()
        tagBaseVec = initL1UniformTags.tagBaseVec
                                    
    process.load('CondTools.L1Trigger.L1CondDBPayloadWriter_cfi')
    
    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
    initPayloadWriter.outputDB = cms.Service("PoolDBOutputService",
                                             CondDBSetup,
                                             connect = cms.string(outputDBConnect),
                                             toPut = cms.VPSet(cms.PSet(
        record = cms.string("L1TriggerKeyListRcd"),
        tag = cms.string("L1TriggerKeyList_" + tagBaseVec[ L1CondEnum.L1TriggerKeyList ]))
                                                               ))
    initPayloadWriter.outputDB.DBParameters.authenticationPath = outputDBAuth
    process.add_(initPayloadWriter.outputDB)
