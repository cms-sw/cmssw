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
        'Geometry/MuonCommonData/data/mf.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml', 
        'Geometry/MuonCommonData/data/muonNumbering.xml', 
        'Geometry/MuonCommonData/data/muonYoke.xml', 
        'Geometry/MuonSimData/data/muonSens.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


