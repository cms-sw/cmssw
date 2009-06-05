import FWCore.ParameterSet.Config as cms

process = cms.Process("check")


# rpc geometry
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")
# emulation

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")
process.load("L1Trigger.RPCTrigger.RPCConeConfig_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff")

process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")


process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.p = cms.EDAnalyzer("RPCConeConnectionsAna",
#       minTower = cms.untracked.int32(-16),
#       maxTower = cms.untracked.int32(16),
#       minSector = cms.untracked.int32(0),
#       maxSector = cms.untracked.int32(11)
       minTower = cms.int32(-16),
       maxTower = cms.int32(16),
       minSector = cms.int32(0),
       maxSector = cms.int32(11)
   
)


process.p1 = cms.Path(process.p)
