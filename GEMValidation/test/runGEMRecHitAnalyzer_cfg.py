import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
# process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
# process.load("Geometry.CMSCommonData.cmsExtendedGeometryPostLS1XML_cfi")
# process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
# process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi')
process.load("Geometry.GEMGeometry.gemGeometry_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

# the analyzer configuration
process.load('RPCGEM.GEMValidation.GEMRecHitAnalyzer_cfi')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    'file:out_localrec.root'
    )
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string("gem_localrec_ana.root")
)

process.p = cms.Path(process.GEMRecHitAnalyzer)
