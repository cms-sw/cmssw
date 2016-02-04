def initIOVWriter( process,
                   outputDBConnect = 'sqlite_file:l1config.db',
                   outputDBAuth = '.',
                   tagBaseVec = [],
                   tscKey = 'dummy' ):
    import FWCore.ParameterSet.Config as cms
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum

    if len( tagBaseVec ) == 0:
        from CondTools.L1Trigger.L1UniformTags_cfi import initL1UniformTags
        initL1UniformTags()
        tagBaseVec = initL1UniformTags.tagBaseVec                                

    process.load('CondTools.L1Trigger.L1CondDBIOVWriter_cfi')
    process.L1CondDBIOVWriter.tscKey = cms.string( tscKey )

    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
    initIOVWriter.outputDB = cms.Service("PoolDBOutputService",
                                         CondDBSetup,
                                         connect = cms.string(outputDBConnect),
                                         toPut = cms.VPSet(cms.PSet(
        record = cms.string("L1TriggerKeyRcd"),
        tag = cms.string("L1TriggerKey_" + tagBaseVec[ L1CondEnum.L1TriggerKey ])),
                                                           cms.PSet(
        record = cms.string("L1TriggerKeyListRcd"),
        tag = cms.string("L1TriggerKeyList_" + tagBaseVec[ L1CondEnum.L1TriggerKeyList ]))
                                                           ))
    initIOVWriter.outputDB.DBParameters.authenticationPath = outputDBAuth

    from CondTools.L1Trigger.L1SubsystemParams_cfi import initL1Subsystems
    initL1Subsystems( tagBaseVec = tagBaseVec )
    initIOVWriter.outputDB.toPut.extend(initL1Subsystems.params.recordInfo)
    process.add_(initIOVWriter.outputDB)
