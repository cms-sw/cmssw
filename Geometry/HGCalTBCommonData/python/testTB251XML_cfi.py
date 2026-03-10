import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalCommonData/data/hgcalMaterial/v3/hgcalMaterial.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/cms.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/caloBase.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcal.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcalcell.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcalwafer.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcalPssive.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcalEE.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcalCons.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcalConsData.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcalsense.xml',
#                               'Geometry/HGCalTBCommonData/data/TB251/CaloUtil.xml',
                               'Geometry/HGCalTBCommonData/data/TB251/hgcProdCuts.xml',
                               ),
    rootNodeName = cms.string('cms:OCMS')
)
