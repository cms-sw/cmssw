def initPayloadWriter( process,
                       outputDBConnect = 'sqlite_file:l1config.db',
                       outputDBAuth = '.',
                       tagBase = 'IDEAL' ):
    import FWCore.ParameterSet.Config as cms
    process.load('CondTools.L1Trigger.L1CondDBPayloadWriter_cfi')
    
    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
    initPayloadWriter.outputDB = cms.Service("PoolDBOutputService",
                                             CondDBSetup,
                                             BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                             connect = cms.string(outputDBConnect),
                                             toPut = cms.VPSet(cms.PSet(
        record = cms.string("L1TriggerKeyListRcd"),
        tag = cms.string("L1TriggerKeyList_" + tagBase))
                                                               ))
    initPayloadWriter.outputDB.DBParameters.authenticationPath = outputDBAuth
    process.add_(initPayloadWriter.outputDB)
