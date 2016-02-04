import FWCore.ParameterSet.Config as cms

process = cms.Process("ecalde")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.test.dqm_onlineEnv_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff")

process.load("DQM.L1TMonitorClient.L1TRPCTFClient_cff")


#process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
#process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")


process.load("DQM.L1TMonitor.L1TRPCTF_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)

#process.source = cms.Source("NewEventStreamFileReader",
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
         '/store/data/CRAFT09/Cosmics/RAW/v1/000/112/417/FE184C1C-DE94-DE11-B6D7-000423D991F0.root'
    )
)

process.gtUnpack = cms.EDFilter("L1GlobalTriggerRawToDigi",
    DaqGtFedId = cms.untracked.int32(813),
    DaqGtInputTag = cms.InputTag("source"),
    UnpackBxInEvent = cms.int32(-1),
    ActiveBoardsMask = cms.uint32(65535)
)

#process.l1trpctf = cms.EDFilter("L1TRPCTF",
#    rpctfRPCDigiSource = cms.InputTag("rpcunpacker"),
#    outputFile = cms.untracked.string(''),
#    verbose = cms.untracked.bool(False),
#    rpctfSource = cms.InputTag("gtUnpack"),
#    MonitorDaemon = cms.untracked.bool(True),
#    DaqMonitorBEInterface = cms.untracked.bool(True)
#)

#process.l1trpctpg = cms.EDFilter("L1TRPCTPG",
#    outputFile = cms.untracked.string(''),
#    verbose = cms.untracked.bool(False),
#    rpctpgSource = cms.InputTag("rpcunpacker"),
#    rpctfSource = cms.InputTag("gtUnpack"),
#    MonitorDaemon = cms.untracked.bool(True),
#    DaqMonitorBEInterface = cms.untracked.bool(True)
#)

#process.l1trpctpg.rpctpgSource = cms.InputTag("rpcunpacker")
#process.l1trpctpg.rpctfSource = cms.InputTag("gtUnpack")



process.l1trpctf.rpctfSource = cms.InputTag("gtUnpack")

process.b = cms.Path(process.gtUnpack)

process.load("DQM.L1TMonitor.L1TRPCTPG_offline_cff")
process.l1trpctpg.rpctpgSource = cms.InputTag("rpcunpacker")
process.l1trpctpg.rpctfSource = cms.InputTag("gtUnpack")

process.a = cms.Path(process.l1trpctf*process.l1trpctfqTester*process.l1trpctfClient*process.dqmEnv*process.dqmSaver)

process.MessageLogger.destinations = ['log.txt']
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'L1T'
process.l1trpctfClient.verbose = True


