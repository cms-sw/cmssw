import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2030/v1/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/coldBox/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/HGCalCommonData/data/coldBox/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/HGCalCommonData/data/coldBox/zm/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v3/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/coldBox/zm/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v19n/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/coldBox/hgcalPassive.xml',
        'Geometry/HGCalCommonData/data/coldBox/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/coldBox/hgcalCons.xml',
        'Geometry/HGCalCommonData/data/coldBox/hgcalConsData.xml',
        'Geometry/HGCalCommonData/data/coldBox/hgcsensv19n.xml',
        'Geometry/HcalSimData/data/CaloUtil/2030/v2c/CaloUtil.xml',
        'Geometry/HGCalCommonData/data/coldBox/hgcProdCutsv15.xml'
    ),
    rootNodeName = cms.string('cms:OCMS')
)
