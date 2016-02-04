import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    moduleLogName = cms.untracked.string('source'),
    fileNames = cms.untracked.vstring(
            '/store/data/Commissioning08/Calo/RAW/v1/000/069/997/00CA7C60-67AD-DD11-AF89-000423D94A20.root'
    )
)


process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
process.load("L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff")
process.load("L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff")
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeomConfig_cff")
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")
process.l1GtUnpack.DaqGtInputTag = cms.InputTag("source")
process.l1GtUnpack.UnpackBxInEvent = cms.int32(-1)



#process.load("L1Trigger.GlobalMuonTrigger.l1GmtEmulDigis_cff")
#process.l1GmtEmulDigis.DTCandidates = cms.InputTag("l1GtUnpack","DT"),
#process.l1GmtEmulDigis.CSCCandidates = cms.InputTag("l1GtUnpack","CSC"),
#process.l1GmtEmulDigis.RPCbCandidates = cms.InputTag("l1GtUnpack","RPCb"),
#process.l1GmtEmulDigis.RPCfCandidates = cms.InputTag("l1GtUnpack","RPCf"),





#process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
# log = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
# debugModules = cms.untracked.vstring("*"),
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


process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

#process.l1RpcEmulDigis.label = cms.string('muonRPCDigis')
process.l1RpcEmulDigis.label = cms.string('rpcunpacker')

#emulator/comparator
process.load("L1Trigger.HardwareValidation.L1HardwareValidation_cff")
#process.load("L1Trigger.Configuration.L1Config_cff import *
process.load("DQM.L1TMonitor.L1TDEMON_cfi")

#process.l1compare.RPCsourceData = cms.InputTag("gtDigis")
process.l1compare.RPCsourceData = cms.InputTag("l1GtUnpack")
process.l1compare.RPCsourceEmul = cms.InputTag("l1RpcEmulDigis")
process.l1compare.COMPARE_COLLS = cms.untracked.vuint32(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
process.l1compare.VerboseFlag   = 0
process.l1compare.DumpMode      = -1

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#DQM = cms.Service("DQM",
#    debug = cms.untracked.bool(False),
#    publishFrequency = cms.untracked.double(5.0),
#    collectorPort = cms.untracked.int32(9090),
#    collectorHost = cms.untracked.string('localhost'),
#    filter = cms.untracked.string('')
#)


process.DQMStore = cms.Service("DQMStore")
#,
#    referenceFileName = cms.untracked.string(''),
#    verbose = cms.untracked.int32(0),
#    collateHistograms = cms.untracked.bool(False)
#)

#process.tdqmSaver = cms.EDFilter("DQMFileSaver",
#     saveAtJobEnd = cms.untracked.bool(False),
#     environment = cms.untracked.string('Offline'),
#     workflow = cms.untracked.string('/A/B/C'),
#     saveAtRunEnd = cms.untracked.bool(False),
#     dirName = cms.untracked.string('.')
#)

process.l1demon.disableROOToutput = cms.untracked.bool(False)


# Time 6178 Orbit start 69802064 orbit end 69813278 triggers: 3323
# Time 6179 Orbit start 69813280 orbit end 69824501 triggers: 7671
# process.flt = cms.EDFilter("OrbitFilter", minOrbit = cms.int32(69802064), maxOrbit = cms.int32(69824501) )
#process.p = cms.Path(process.flt*process.rpcunpacker*process.l1RpcEmulDigis*process.l1GtUnpack*process.l1compare*process.l1demon)

process.p = cms.Path(process.rpcunpacker*process.l1RpcEmulDigis*process.l1GtUnpack*process.l1compare*process.l1demon)




