import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalTBCommonData/data/TB161/cms.xml',
        'Geometry/HGCalTBCommonData/data/TB161/hgcal.xml',
        'Geometry/HGCalTBCommonData/data/TB161/8ModuleV2/hgcalEE.xml',
        'Geometry/HGCalTBCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
        'Geometry/HGCalTBCommonData/data/TB161/hgcalBeam.xml',
        'Geometry/HGCalTBCommonData/data/TB161/hgcalsense.xml',
        'Geometry/HGCalTBCommonData/data/TB161/hgcProdCuts.xml',
        'Geometry/HGCalTBCommonData/data/TB161/8Module/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


