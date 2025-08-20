import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v1/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v3/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v19/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v19/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v19/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v19/hgcalpos.xml'),
    rootNodeName = cms.string('cms:OCMS')
)
