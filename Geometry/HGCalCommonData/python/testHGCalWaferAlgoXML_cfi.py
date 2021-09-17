import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalpos.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml'),
    rootNodeName = cms.string('cms:OCMS')
)
