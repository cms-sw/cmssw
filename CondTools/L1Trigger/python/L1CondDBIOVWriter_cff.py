def initIOVWriter( process,
                   outputDBConnect = 'sqlite_file:l1config.db',
                   outputDBAuth = '.',
                   tagBase = 'IDEAL',
                   tscKey = 'dummy' ):
    import FWCore.ParameterSet.Config as cms
    process.load('CondTools.L1Trigger.L1CondDBIOVWriter_cfi')
    process.L1CondDBIOVWriter.tscKey = cms.string( tscKey )

    from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
    initIOVWriter.outputDB = cms.Service("PoolDBOutputService",
                                         CondDBSetup,
                                         BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                         connect = cms.string(outputDBConnect),
                                         toPut = cms.VPSet(cms.PSet(
        record = cms.string("L1TriggerKeyRcd"),
        tag = cms.string("L1TriggerKey_" + tagBase))
                                                           ))
    initIOVWriter.outputDB.DBParameters.authenticationPath = outputDBAuth

    from CondTools.L1Trigger.L1SubsystemParams_cfi import initL1Subsystems
    initL1Subsystems( tagBase = tagBase )
    initIOVWriter.outputDB.toPut.extend(initL1Subsystems.params.recordInfo)
    process.add_(initIOVWriter.outputDB)
