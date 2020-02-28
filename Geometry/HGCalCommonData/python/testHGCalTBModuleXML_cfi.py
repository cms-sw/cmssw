import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/TB170/cms.xml',
        'Geometry/HGCalCommonData/data/TB170/hgcal.xml',
        'Geometry/HGCalCommonData/data/TB170/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


