def initIOVWriterExt( process,
                   outputDBConnect = 'sqlite_file:l1config.db',
                   outputDBAuth = '.',
                   tagBaseVec = [],
                   tscKey = 'dummy',
                   rsKey  = 'dummy' ):
    import FWCore.ParameterSet.Config as cms
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt

    if len( tagBaseVec ) == 0:
        from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
        initL1UniformTagsExt()
        tagBaseVec = initL1UniformTagsExt.tagBaseVec                                

    process.load('CondTools.L1TriggerExt.L1CondDBIOVWriterExt_cfi')
    process.L1CondDBIOVWriterExt.tscKey = cms.string( tscKey )
    process.L1CondDBIOVWriterExt.rsKey  = cms.string( rsKey )

    from CondCore.CondDB.CondDB_cfi import CondDB
    CondDB.connect = cms.string(outputDBConnect)

    initIOVWriterExt.outputDB = cms.Service("PoolDBOutputService",
                                         CondDB,
                                         toPut = cms.VPSet(cms.PSet(
        record = cms.string("L1TriggerKeyExtRcd"),
        tag = cms.string("L1TriggerKeyExt_" + tagBaseVec[ L1CondEnumExt.L1TriggerKeyExt ])),
                                                           cms.PSet(
        record = cms.string("L1TriggerKeyListExtRcd"),
        tag = cms.string("L1TriggerKeyListExt_" + tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ]))
                                                           ))
    initIOVWriterExt.outputDB.DBParameters.authenticationPath = outputDBAuth

    from CondTools.L1TriggerExt.L1SubsystemParamsExt_cfi import initL1SubsystemsExt
    initL1SubsystemsExt( tagBaseVec = tagBaseVec )
    initIOVWriterExt.outputDB.toPut.extend(initL1SubsystemsExt.params.recordInfo)
    process.add_(initIOVWriterExt.outputDB)
