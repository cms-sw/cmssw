def initPayloadWriterExt( process,
                       outputDBConnect = 'sqlite_file:l1config.db',
                       outputDBAuth = '.',
                       tagBaseVec = [] ):
    import FWCore.ParameterSet.Config as cms
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt

    if len( tagBaseVec ) == 0:
        from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
        initL1UniformTagsExt()
        tagBaseVec = initL1UniformTagsExt.tagBaseVec
                                    
    process.load('CondTools.L1TriggerExt.L1CondDBPayloadWriterExt_cfi')
    
    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
    initPayloadWriterExt.outputDB = cms.Service("PoolDBOutputService",
                                             CondDBSetup,
                                             connect = cms.string(outputDBConnect),
                                             toPut = cms.VPSet(cms.PSet(
        record = cms.string("L1TriggerKeyListExtRcd"),
        tag = cms.string("L1TriggerKeyListExt_" + tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ]))
                                                               ))
    initPayloadWriterExt.outputDB.DBParameters.authenticationPath = outputDBAuth
    process.add_(initPayloadWriterExt.outputDB)
