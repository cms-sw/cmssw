import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/EcalTestBeam/data/ecal_TB.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/EcalSimData/data/ecalsens_MTCC.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'Geometry/EcalSimData/data/EcalProdCuts.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


