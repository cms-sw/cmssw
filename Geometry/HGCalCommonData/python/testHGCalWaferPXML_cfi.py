import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v1/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v16p/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v16/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v16p/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v16p/hgcalpos.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


