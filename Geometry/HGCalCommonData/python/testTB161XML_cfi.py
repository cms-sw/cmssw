import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/TB161/cms.xml',
        'Geometry/HGCalCommonData/data/TB161/hgcal.xml',
        'Geometry/HGCalCommonData/data/TB161/8ModuleV2/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/v7/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/TB161/hgcalBeam.xml',
        'Geometry/HGCalCommonData/data/TB161/hgcalsense.xml',
        'Geometry/HGCalCommonData/data/TB161/hgcProdCuts.xml',
        'Geometry/HGCalCommonData/data/TB161/8Module/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


