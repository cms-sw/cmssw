import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalCommonData/data/hgcalMaterial/v3/hgcalMaterial.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/cms.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/caloBase.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcal.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcalcell.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcalwafer.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcalEE.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcalHE.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcalCons.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcalConsData.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcalsense.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/CaloUtil.xml',
                               'Geometry/HGCalTBCommonData/data/TB231/hgcProdCuts.xml',
                               ),
    rootNodeName = cms.string('cms:OCMS')
)


