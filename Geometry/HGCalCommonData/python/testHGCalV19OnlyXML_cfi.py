import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v1/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v1/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/cms/2030/v0/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/caloBase/2030/v7/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v3/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcal/v19/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v19/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v19/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalPassive/v19/hgcalPassive.xml',
        'Geometry/HGCalCommonData/data/hgcalEE/v19/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalHEsil/v19/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/hgcalHEmix/v19/hgcalHEmix.xml',
        'Geometry/HGCalCommonData/data/hgcalCons/v19/hgcalCons.xml',
        'Geometry/HGCalCommonData/data/hgcalConsData/v19/hgcalConsData.xml',
        'Geometry/HcalSimData/data/CaloUtil/2030/v2c/CaloUtil.xml',
        'Geometry/HGCalSimData/data/hgcsensv15.xml',
        'Geometry/HGCalSimData/data/hgcProdCuts/2030/v1/HcalProdCuts.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml'
    ),
    rootNodeName = cms.string('cms:OCMS')
)
