import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register("runNumber",
                 326479,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "")
options.register("maxEvents",
                 100,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events to run")
options.parseArguments()

process = cms.Process("DQM")

process.load("CondCore.CondDB.CondDB_cfi")
process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDB,
                                        DumpStat = cms.untracked.bool(True),
                                        toGet = cms.VPSet(
                                            cms.PSet(
                                                record = cms.string('BeamSpotOnlineLegacyObjectsRcd'),
                                                tag = cms.string("BeamSpotOnlineLegacy"),
                                                refreshTime = cms.uint64(2)
                                            ),
                                            cms.PSet(
                                                record = cms.string('BeamSpotOnlineHLTObjectsRcd'),
                                                tag = cms.string("BeamSpotOnlineHLT"),
                                                refreshTime = cms.uint64(2)

                                            ),
                                        ),
                                        )
process.BeamSpotDBSource.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS') 

import RecoVertex.BeamSpotProducer.onlineBeamSpotESProducer_cfi as _mod
process.BeamSpotESProducer = _mod.onlineBeamSpotESProducer.clone(
        timeThreshold = 999999 # for express allow >48h old payloads for replays. DO NOT CHANGE
)

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)
# process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))

process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(1)
    ),
    OnlineBeamMonitor = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1), # every 1000th only
 	limit = cms.untracked.int32(0)
    ))

process.source = cms.Source("EmptySource")
process.source.numberEventsInRun=cms.untracked.uint32(100)
process.source.firstRun = cms.untracked.uint32(options.runNumber)
process.source.firstLuminosityBlock = cms.untracked.uint32(1)
process.source.numberEventsInLuminosityBlock = cms.untracked.uint32(1)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

#process.load("DQMServices.Core.DQMEDAnalyzer") 
process.onlineBeamMonitor = cms.EDProducer("OnlineBeamMonitor",
                                           MonitorName         = cms.untracked.string("onlineBeamMonitor"),
                                           AppendRunToFileName = cms.untracked.bool(False),
                                           WriteDIPAscii       = cms.untracked.bool(True),
                                           DIPFileName         = cms.untracked.string("BeamFitResultsForDIP.txt"))


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

#process.Tracer = cms.Service("Tracer")

process.dqmcommon = cms.Sequence(process.dqmEnv*process.dqmSaver*process.dqmSaverPB)
process.pp = cms.Path(process.onlineBeamMonitor+process.dqmcommon)
process.schedule = cms.Schedule(process.pp)
