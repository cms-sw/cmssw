import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generate2026Geometry.py
# If you notice a mistake, please update the generating script, not just this config

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms/2026/v2/cms.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/caloBase/2026/v2/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/CMSCommonData/data/beampipe/2026/v1/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam/2026/v1/cmsBeam.xml',
        'Geometry/CMSCommonData/data/cavernData/2017/v1/cavernData.xml',
        'Geometry/EcalCommonData/data/eregalgo/2026/v2/eregalgo.xml',
        'Geometry/EcalCommonData/data/ectkcable/2026/v1/ectkcable.xml',
        'Geometry/EcalCommonData/data/ectkcablemat/2026/v1/ectkcablemat.xml',
        'Geometry/HcalCommonData/data/hcalrotations.xml',
        'Geometry/HcalCommonData/data/hcal/v2/hcalalgo.xml',
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml',
        'Geometry/HcalCommonData/data/hcalcablealgo/v2/hcalcablealgo.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcal/v10/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalEE/v10/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalHEsil/v11/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/hgcalHEmix/v11/hgcalHEmix.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v9/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v9/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalCons/v11/hgcalCons.xml',
    )+
    cms.vstring(
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)
