import FWCore.ParameterSet.Config as cms

process = cms.Process("check")


# rpc geometry
process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/muonBase.xml',
        'Geometry/CMSCommonData/data/cmsMuon.xml',
        'Geometry/CMSCommonData/data/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam.xml',
        'Geometry/CMSCommonData/data/mgnt.xml',
        'Geometry/CMSCommonData/data/muonMagnet.xml',
        'Geometry/CMSCommonData/data/cavern.xml',
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/muonMB.xml',
        'Geometry/MuonCommonData/data/mbCommon.xml',
        'Geometry/MuonCommonData/data/mb1.xml',
        'Geometry/MuonCommonData/data/mb2.xml',
        'Geometry/MuonCommonData/data/mb3.xml',
        'Geometry/MuonCommonData/data/mb4.xml',
        'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml',
        'Geometry/MuonCommonData/data/mf_upscope.xml',
      #  'Geometry/MuonCommonData/data/mf.xml',
        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml',
        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml',
        'Geometry/RPCGeometryBuilder/data/RPCSpecs.xml',
        'Geometry/MuonCommonData/data/muonNumbering_upscope.xml',
     #   'Geometry/MuonCommonData/data/muonNumbering.xml',
        'Geometry/MuonCommonData/data/muonYoke_upscope.xml',
      #  'Geometry/MuonCommonData/data/muonYoke.xml',
        'Geometry/MuonSimData/data/muonSens.xml'),
    rootNodeName = cms.string('cms:OCMS')
)



process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
#process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("L1TriggerConfig.RPCTriggerConfig.RPCConeDefinition_cff")
# emulation

process.load("L1TriggerConfig.RPCTriggerConfig.L1RPCConfig_cff")
process.load("L1Trigger.RPCTrigger.RPCConeConfig_cff")
process.load("L1TriggerConfig.RPCTriggerConfig.RPCHwConfig_cff")

process.load("EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi")
process.RPCCabling.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_30X_RPC')

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
