import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
    moduleLogName = cms.untracked.string('source'),
    fileNames = cms.untracked.vstring(
            '/store/data/Summer08/Cosmics/RECO/CRAFT_ALL_V8_v1/0000/047AADFF-0FF4-DD11-B47F-001D0968F0B2.root'
    )
)



process.MessageLogger = cms.Service("MessageLogger",
 destinations = cms.untracked.vstring('log.txt')
)

# rpc geometry
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

# emulation
process.load("L1TriggerConfig.RPCTriggerConfig.RPCPatSource_cfi")

# Choose proper patterns here!
process.rpcconf.filedir = cms.untracked.string('MyAna/DataEmuComp/data/CosmicPats6/')
process.rpcconf.PACsPerTower = cms.untracked.int32(1)


process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeSource_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfigSource_cfi")
process.load("L1Trigger.RPCTrigger.l1RpcEmulDigis_cfi")

process.l1RpcEmulDigis.label = cms.string('muonRPCDigis')

process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")
process.load("DQM.L1TMonitor.L1TDEMON_cfi")

process.l1compare.RPCsourceData = cms.InputTag("gtDigis")
process.l1compare.RPCsourceEmul = cms.InputTag("l1RpcEmulDigis")


process.l1compare.COMPARE_COLLS = cms.untracked.vuint32(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
process.l1compare.VerboseFlag   = 0
process.l1compare.DumpMode      = -1

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.DQMStore = cms.Service("DQMStore")

process.l1demon.disableROOToutput = cms.untracked.bool(False)


#process.p = cms.Path(process.l1RpcEmulDigis*process.l1GtUnpack*process.l1compare*process.l1demon)
process.p = cms.Path(process.l1RpcEmulDigis*process.l1compare*process.l1demon)


