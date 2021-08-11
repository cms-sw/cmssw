import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v1/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15f/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v15/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15f/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v15f/hgcalpos.xml',
        'Geometry/HGCalCommonData/data/hgcalCons/v15f/hgcalCons.xml',
        'Geometry/HGCalCommonData/data/hgcalConsData/v15f/hgcalConsData.xml',
        'Geometry/HGCalSimData/data/hgcsensEE.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


