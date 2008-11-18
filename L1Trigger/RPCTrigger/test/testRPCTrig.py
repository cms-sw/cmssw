import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")


#process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    log = cms.untracked.PSet( threshold = cms.untracked.string("DEBUG") ),
    debugModules = cms.untracked.vstring("l1RpcEmulDigis"),
    destinations = cms.untracked.vstring('log')
)

# rpc geometry
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

# emulation
process.load("L1TriggerConfig.RPCTriggerConfig.RPCPatSource_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeSource_cfi")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfigSource_cfi")
process.load("L1Trigger.RPCTrigger.l1RpcEmulDigis_cfi")
process.l1RpcEmulDigis.label = cms.string('rpcunpacker')

# rpc r2d
process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)

process.source = cms.Source("NewEventStreamFileReader",
#process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

            '/store/data/GlobalCruzet4/A/000/057/620/GlobalCruzet4.00057620.0001.A.storageManager.1.0000.dat'

    )
)

process.a = cms.Path(process.rpcunpacker*process.l1RpcEmulDigis)
