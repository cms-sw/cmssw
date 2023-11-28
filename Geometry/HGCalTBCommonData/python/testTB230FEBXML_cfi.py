import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalCommonData/data/hgcalMaterial/v2/hgcalMaterial.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/cms.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/hgcal.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/hgcalcell.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/hgcalwafer.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/hgcalEE.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/hgcalCons.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/hgcalConsData.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/FEB230/hgcalsense.xml',
                               'Geometry/HGCalTBCommonData/data/TB230/hgcProdCuts.xml',
                               ),
    rootNodeName = cms.string('cms:OCMS')
)


