import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/ForwardCommonData/data/hfnoseCell/v1/hfnose.xml',
        'Geometry/ForwardCommonData/data/hfnoseCell/v1/hfnoseCell.xml'),
    rootNodeName = cms.string('cms:OCMS')
)
