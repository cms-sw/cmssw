import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.load("CondCore.CondDB.CondDB_cfi")
process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDB,
                                        toGet = cms.VPSet(
                                            cms.PSet(
                                                record = cms.string('BeamSpotOnlineLegacyObjectsRcd'),
                                                tag = cms.string("BeamSpotOnlineTestLegacy"),
                                                refreshTime = cms.uint64(1)
                                            ),
                                            cms.PSet(
                                                record = cms.string('BeamSpotOnlineHLTObjectsRcd'),
                                                tag = cms.string("BeamSpotOnlineTestHLT"),
                                                refreshTime = cms.uint64(1)

                                            ),
                                ),
                                        )
process.BeamSpotDBSource.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS') 
process.BeamSpotESProducer = cms.ESProducer("OnlineBeamSpotESProducer")

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(0)
    ),
    OnlineBeamMonitor = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1), # every 1000th only
	limit = cms.untracked.int32(0)
    )
)
#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)
process.source = cms.Source("EmptySource")
process.source.numberEventsInRun=cms.untracked.uint32(20)
process.source.firstRun = cms.untracked.uint32(336055)
process.source.firstLuminosityBlock = cms.untracked.uint32(49)
process.source.numberEventsInLuminosityBlock = cms.untracked.uint32(1)
process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(100)
)

#process.load("DQMServices.Core.DQMEDAnalyzer") 
process.onlineBeamMonitor = cms.EDProducer("OnlineBeamMonitor",
MonitorName = cms.untracked.string("onlineBeamMonitor"))


# DQM Live Environment
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.tag           = 'BeamMonitor'

process.dqmEnvPixelLess = process.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'



#import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
#process.offlineBeamSpotForDQM = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32 (4),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(2)

    )

process.pp = cms.Path(process.onlineBeamMonitor+process.dqmSaver)
process.schedule = cms.Schedule(process.pp)
