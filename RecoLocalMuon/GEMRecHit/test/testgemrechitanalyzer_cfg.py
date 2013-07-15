import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
# process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
# process.load("Geometry.CMSCommonData.cmsExtendedGeometryPostLS1XML_cfi")
#  8 eta partitions :: command line option :: --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi.py
# 10 eta partitions :: command line option :: --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi.py
# process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMXML_cfi')
process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
# process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr10v01XML_cfi')
process.load("Geometry.GEMGeometry.gemGeometry_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        # 'file:/afs/cern.ch/user/p/piet/work/Analysis/CMSSW_6_0_1_PostLS1v1/src/cmsDriverCommands/SingleMuPt100_cfi_RECO.root'
        # 'file:/afs/cern.ch/user/p/piet/public/RPCRootFiles/SingleMuPt100_cfi_RECO_V12.root'
        # 'file:/afs/cern.ch/user/p/piet/public/RPCRootFiles/SingleMuPt100_cfi_RECO_V12_25evt.root'
        'file:/afs/cern.ch/user/p/piet/work/Analysis/CMSSW_6_0_1_PostLS1v2_patch4/src/cmsRunConfigFiles/SingleMuPt40_RECHIT.root'
    )
)

process.demo = cms.EDAnalyzer('TestGEMRecHitAnalyzer',
                              RootFileName = cms.untracked.string("TestGEMRecHitHistograms.root"),

)


process.p = cms.Path(process.demo)
