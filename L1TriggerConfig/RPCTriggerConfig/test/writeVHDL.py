import FWCore.ParameterSet.Config as cms

process = cms.Process("writeVHD")




process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")
#process.rpcconf.filedir = cms.untracked.string('L1TriggerConfig/RPCTriggerConfig/test/v5/')
process.rpcconf.filedir = cms.untracked.string('L1Trigger/RPCPatts/data/D_20110921_fixedCones_new36__all_12/')

    
process.rpcconf.PACsPerTower = cms.untracked.int32(12)

process.load("L1Trigger.RPCTrigger.RPCConeConfig_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag = 'START42_V13::All'
process.GlobalTag.globaltag = cms.string('GR_R_42_V20::All')

process.es_prefer_rpcPats = cms.ESPrefer("RPCTriggerConfig","rpcconf") 
# PoolDBESSource" label="RPCCabling"

#process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")

#process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'sqlite_file:RPCEMap3.db'

'''
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.RPCCabling = cms.ESSource("PoolDBESSource",
     process.CondDBSetup,
     toGet = cms.VPSet(cms.PSet(
         record = cms.string('RPCEMapRcd'),
         tag = cms.string('RPCEMap_v2')
     )),
     connect = cms.string('sqlite_file:RPCEMap3.db')
)
process.es_prefer_cabling = cms.ESPrefer("PoolDBESSource","RPCCabling") 
'''


process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(175906)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.write = cms.EDAnalyzer("WriteVHDL",
          minTower = cms.int32(-12),
          maxTower = cms.int32(12),
          minSector = cms.int32(0),
          maxSector = cms.int32(11),
          templateName = cms.string("pacTemplate.vhd"),
          outDir =  cms.string("/tmp/tfruboes/cones/")
)


process.p1 = cms.Path(process.write)
