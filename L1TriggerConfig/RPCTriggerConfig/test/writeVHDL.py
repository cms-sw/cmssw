import FWCore.ParameterSet.Config as cms

process = cms.Process("writeVHD")




process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeConfig_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff")


process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.write = cms.EDAnalyzer("WriteVHDL",
          minTower = cms.int32(0),
          maxTower = cms.int32(0),
          minSector = cms.int32(0),
          maxSector = cms.int32(0),
          templateName = cms.string("pacTemplate.vhd"),
          outDir =  cms.string("./")
)


process.p1 = cms.Path(process.write)
