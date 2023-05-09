import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalTBCommonData/data/TB160/cms.xml',
        'Geometry/HGCalTBCommonData/data/TB160/hgcal.xml',
        'Geometry/HGCalTBCommonData/data/TB160/16Module/hgcalEE.xml',
        'Geometry/HGCalTBCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
        'Geometry/HGCalTBCommonData/data/TB160/hgcalBeam.xml',
        'Geometry/HGCalTBCommonData/data/TB160/hgcalsense.xml',
        'Geometry/HGCalTBCommonData/data/TB160/hgcProdCuts.xml',
        'Geometry/HGCalTBCommonData/data/TB160/16Module/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


