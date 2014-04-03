import FWCore.ParameterSet.Config as cms

#
#  This cfi should be included to parse the muon standalone XML geometry.
#
XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
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
        'Geometry/MuonCommonData/data/RPC2Gap/mbCommon.xml', 
        'Geometry/MuonCommonData/data/RPC2Gap/mb1.xml', 
        'Geometry/MuonCommonData/data/RPC2Gap/mb2.xml', 
        'Geometry/MuonCommonData/data/RPC2Gap/mb3.xml', 
        'Geometry/MuonCommonData/data/RPC2Gap/mb4.xml', 
        'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml', 
        'Geometry/MuonCommonData/data/v2/mf.xml',
        'Geometry/MuonCommonData/data/RPC2Gap/rpcf.xml',                                
        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml', 
        'Geometry/RPCGeometryBuilder/data/RPCSpecs.xml', 
        'Geometry/MuonCommonData/data/RPC2Gap/muonNumbering.xml', 
        'Geometry/MuonCommonData/data/muonYoke.xml', 
        'Geometry/MuonSimData/data/muonSens.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


