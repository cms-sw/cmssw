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

process.load("DQM.L1TMonitor.L1TRPCTF_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.source = cms.Source("NewEventStreamFileReader",
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#        "/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_V4P_StreamALCARECORpcCalHLT_v12/0024/36660505-DBAD-DD11-BB9A-000423D9517C.root"
         "/store/data/Commissioning08/Cosmics/ALCARECO/CRAFT_V4P_StreamALCARECORpcCalHLT_v12/0024/1A9E5171-98AD-DD11-8ADE-001617C3B6FE.root"
    )
)

process.l1trpctf.rpctfSource = cms.InputTag("gtDigis")
process.l1trpctf.rateUpdateTime = cms.int32(-1) # -1 update at end of run



# include this later - contains path with rpc unpacker
#process.load("DQM.L1TMonitor.L1TRPCTPG_offline_cff")
#process.l1trpctpg.rpctpgSource = cms.InputTag("rpcunpacker")
#process.l1trpctpg.rpctfSource = cms.InputTag("gtUnpack")

process.a = cms.Path(process.l1trpctf*process.l1trpctfqTester*process.l1trpctfClient*process.dqmEnv*process.dqmSaver)

process.MessageLogger.destinations = ['log.txt']
process.dqmSaver.convention = 'Online'
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'L1T'
process.l1trpctfClient.verbose = True


