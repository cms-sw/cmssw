import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/CMSCommonData/data/extend/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMother.xml', 
        'Geometry/FP420CommonData/data/cmsfp420.xml', 
        'Geometry/FP420CommonData/data/fp420.xml', 
        'Geometry/FP420CommonData/data/zzzrectangle.xml', 
        'Geometry/FP420CommonData/data/materialsfp420.xml', 
        'Geometry/FP420CommonData/data/FP420Rot.xml', 
        'Geometry/FP420SimData/data/fp420sens.xml', 
        'Geometry/FP420SimData/data/FP420ProdCuts.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


