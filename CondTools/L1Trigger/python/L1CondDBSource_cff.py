def initCondDBSource( process,
                      inputDBConnect = 'frontier://FrontierProd/CMS_COND_31X_L1T',
                      inputDBAuth = '.',
                      tagBaseVec = [],
                      includeAllTags = False,
                      includeRSTags = False,
                      use30XTagList = False,
                      applyESPrefer = True ):
    import FWCore.ParameterSet.Config as cms
    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
    from CondTools.L1Trigger.L1CondEnum_cfi import L1CondEnum

    if len( tagBaseVec ) == 0:
        from CondTools.L1Trigger.L1O2OTags_cfi import initL1O2OTags
        initL1O2OTags()
        tagBaseVec = initL1O2OTags.tagBaseVec
                                
    process.l1conddb = cms.ESSource("PoolDBESSource",
                            CondDBSetup,
                            connect = cms.string(inputDBConnect),
                            toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListRcd'),
        tag = cms.string('L1TriggerKeyList_' + tagBaseVec[ L1CondEnum.L1TriggerKeyList ])
        ),
                                              cms.PSet(
        record = cms.string('L1TriggerKeyRcd'),
        tag = cms.string('L1TriggerKey_' + tagBaseVec[ L1CondEnum.L1TriggerKey ])
        ))
                                    )
    process.l1conddb.DBParameters.authenticationPath = inputDBAuth

    # The more records, the longer it takes PoolDBESSource to initialize, so be
    # selective if possible.

    if includeAllTags == True:
        if use30XTagList == True:
            from CondTools.L1Trigger.L1SubsystemParams30X_cfi import initL1Subsystems
        else:
            from CondTools.L1Trigger.L1SubsystemParams_cfi import initL1Subsystems
        initL1Subsystems( tagBaseVec = tagBaseVec )
        process.l1conddb.toGet.extend(initL1Subsystems.params.recordInfo)
    elif includeRSTags == True:
        from CondTools.L1Trigger.L1RSSubsystemParams_cfi import initL1RSSubsystems
        initL1RSSubsystems( tagBaseVec = tagBaseVec )
        process.l1conddb.toGet.extend(initL1RSSubsystems.params.recordInfo)

    if applyESPrefer == True:
        process.es_prefer_l1conddb = cms.ESPrefer("PoolDBESSource","l1conddb")
