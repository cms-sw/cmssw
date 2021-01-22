import FWCore.ParameterSet.Config as cms

process = cms.Process("rpctest")


#process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('rpcTriggerDigis'),
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

useGlobalTag = 'IDEAL_31X'
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = useGlobalTag + '::All'

# emulation
process.load("L1Trigger.RPCTrigger.rpcTriggerDigis_cff")
process.rpcTriggerDigis.label = cms.string('simMuonRPCDigis')

# rpc r2d
#process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
#process.load("EventFilter.RPCRawToDigi.rpcUnpacker_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)

#process.source = cms.Source("NewEventStreamFileReader",
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

            'file:digi.root'

    )
)

#process.a = cms.Path(process.rpcunpacker*process.l1RpcEmulDigis)
process.a = cms.Path(process.rpcTriggerDigis)
