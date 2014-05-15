import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
         'Geometry/CMSCommonData/data/rotations.xml',
         'Geometry/HGCalCommonData/data/shashlikcapsule.xml'),
    rootNodeName = cms.string('shashlikcapsule:ShashlikCapsule')
)

 
