import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15/hgcalpos.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v9/hgcalcell.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


