import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generate2026Geometry.py
# If you notice a mistake, please update the generating script, not just this config

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v1/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/cms/2026/v5/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/caloBase/2026/v7/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v2/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcal/v18/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v17/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v18/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalPassive/v18/hgcalPassive.xml',
        'Geometry/HGCalCommonData/data/hgcalEE/v18/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalHEsil/v18/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/hgcalHEmix/v18/hgcalHEmix.xml',
        'Geometry/HGCalCommonData/data/hgcalCons/v18/hgcalCons.xml',
        'Geometry/HGCalCommonData/data/hgcalConsData/v17/hgcalConsData.xml',
        'Geometry/HcalSimData/data/CaloUtil/2026/v2c/CaloUtil.xml',
        'Geometry/HGCalSimData/data/hgcsensv15.xml',
        'Geometry/HcalSimData/data/CaloUtil/2026/v2c/CaloUtil.xml',
        'Geometry/HGCalSimData/data/hgcProdCutsv15.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)
