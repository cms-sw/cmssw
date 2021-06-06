import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v1/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15p/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v15/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15p/hgcalpos.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15p/hgcalwafer.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


