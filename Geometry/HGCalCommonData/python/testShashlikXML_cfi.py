import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
         'Geometry/CMSCommonData/data/rotations.xml',
         'Geometry/CMSCommonData/data/extend/cmsextent.xml',
         'Geometry/CMSCommonData/data/cms.xml',
         'Geometry/CMSCommonData/data/cmsMother.xml',
         'Geometry/CMSCommonData/data/eta3/etaMax.xml',
         'Geometry/CMSCommonData/data/PhaseII/caloBase.xml',
         'Geometry/CMSCommonData/data/cmsCalo.xml',
         'Geometry/HGCalCommonData/data/shashlik.xml',
         'Geometry/HGCalCommonData/data/shashliksupermodule.xml',
         'Geometry/HGCalCommonData/data/shashlikmodule.xml',
         'Geometry/HGCalCommonData/data/shashlikConstEta3.xml',
         'Geometry/HGCalSimData/data/shashliksens.xml',
         'Geometry/HGCalSimData/data/shashlikProdCuts.xml'),
    rootNodeName = cms.string('cms:OCMS')
)

 
