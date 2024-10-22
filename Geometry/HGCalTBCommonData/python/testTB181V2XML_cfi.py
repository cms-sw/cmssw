import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct181/cms.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct181/hgcal.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Test/hgcalHE.xml',
                               'Geometry/HGCalTBCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Test/hgcalsense.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Test/hgcProdCuts.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Test/hgcalCons.xml',
                               'Geometry/HcalSimData/data/CaloUtil/2026/v5c/CaloUtil.xml'
 
                               ),
    rootNodeName = cms.string('cms:OCMS')
)
