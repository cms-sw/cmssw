import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")


#process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('l1RpcEmulDigis'),
    files = cms.untracked.PSet(
        log = cms.untracked.PSet(
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

# rpc geometry
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")
# emulation
process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")

process.load("L1Trigger.RPCTrigger.RPCConeConfig_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCBxOrConfig_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHsbConfig_cff")

                                       #TC  11  0  1  2  3  4  5  6
process.l1RPCHsbConfig.hsb0Mask = cms.vint32(3, 3, 3, 3, 3, 3, 3, 3)
                                       #TC   5  6  7  8  9 10 11  0
process.l1RPCHsbConfig.hsb1Mask = cms.vint32(3, 3, 3, 3, 3, 3, 3, 3)

#process.l1RPCHsbConfig.hsb0Mask = cms.vint32(0, 0, 0, 0 , 0, 0 , 0 ,0 )
#process.l1RPCHsbConfig.hsb1Mask = cms.vint32(0, 0, 0, 0 , 0, 0 , 0 ,0 )


process.load("L1Trigger.RPCTrigger.l1RpcEmulDigis_cfi")
process.l1RpcEmulDigis.label = cms.string('simMuonRPCDigis')
process.l1RpcEmulDigis.RPCTriggerDebug = 1

# rpc r2d
#process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
#process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.source = cms.Source("NewEventStreamFileReader",
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

            'file:/tmp/fruboes/SingleMuPt10_cfi_py_GEN_SIM_DIGI.root'

    )
)

process.a = cms.Path(process.l1RpcEmulDigis)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('l1.root')
)

#process.this_is_the_end = cms.EndPath(process.out)

