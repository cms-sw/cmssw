# Default behavior: deliver current L1-O2O tags from Frontier.

# IF tagBaseVec is empty then
#    IF tagBaseVec is given, use uniform tags with the given tagBase
#    ELSE use current L1-O2O tags

# If tagBase AND tagBaseVec are both given, then tagBaseVec trumps tagBase.

def initCondDBSourceExt( process,
                      inputDBConnect = 'frontier://FrontierProd/CMS_CONDITIONS',
                      inputDBAuth = '.',
                      tagBase = "",
                      tagBaseVec = [],
                      includeAllTags = False,
                      includeRSTags = False,
                      applyESPrefer = True ):
    import FWCore.ParameterSet.Config as cms
    from CondTools.L1TriggerExt.L1CondEnumExt_cfi import L1CondEnumExt

    if len( tagBaseVec ) == 0:
        if len( tagBase ) != 0:
            from CondTools.L1TriggerExt.L1UniformTagsExt_cfi import initL1UniformTagsExt
            initL1UniformTagsExt( tagBase )
            tagBaseVec = initL1UniformTagsExt.tagBaseVec
        else:
            from CondTools.L1TriggerExt.L1O2OTagsExt_cfi import initL1O2OTagsExt
            initL1O2OTagsExt()
            tagBaseVec = initL1O2OTagsExt.tagBaseVec

    from CondCore.CondDB.CondDB_cfi import CondDB
    CondDB.connect = cms.string(inputDBConnect)

    process.l1conddb = cms.ESSource("PoolDBESSource",
                            CondDB,
                            toGet = cms.VPSet(cms.PSet(
        record = cms.string('L1TriggerKeyListExtRcd'),
        tag = cms.string('L1TriggerKeyListExt_' + tagBaseVec[ L1CondEnumExt.L1TriggerKeyListExt ])
        ),
                                              cms.PSet(
        record = cms.string('L1TriggerKeyExtRcd'),
        tag = cms.string('L1TriggerKeyExt_' + tagBaseVec[ L1CondEnumExt.L1TriggerKeyExt ])
        ))
                                    )
    process.l1conddb.DBParameters.authenticationPath = inputDBAuth

    # The more records, the longer it takes PoolDBESSource to initialize, so be
    # selective if possible.

    if includeAllTags == True:
        from CondTools.L1TriggerExt.L1SubsystemParamsExt_cfi import initL1SubsystemsExt
        initL1SubsystemsExt( tagBaseVec = tagBaseVec )
        process.l1conddb.toGet.extend(initL1SubsystemsExt.params.recordInfo)
    elif includeRSTags == True:
        from CondTools.L1TriggerExt.L1RSSubsystemParamsExt_cfi import initL1RSSubsystemsExt
        initL1RSSubsystemsExt( tagBaseVec = tagBaseVec )
        process.l1conddb.toGet.extend(initL1RSSubsystemsExt.params.recordInfo)

    if applyESPrefer == True:
        process.es_prefer_l1conddb = cms.ESPrefer("PoolDBESSource","l1conddb")
