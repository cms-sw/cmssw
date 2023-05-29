import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v1/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v2/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v16/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v16/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v16/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v16/hgcalpos.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


