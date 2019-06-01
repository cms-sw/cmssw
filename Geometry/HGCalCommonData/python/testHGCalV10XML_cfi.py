import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generate2023Geometry.py
# If you notice a mistake, please update the generating script, not just this config

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms/2023/v2/cms.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/caloBase/2023/v2/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/CMSCommonData/data/beampipe/2023/v1/beampipe.xml',
        'Geometry/CMSCommonData/data/cmsBeam/2023/v1/cmsBeam.xml',
        'Geometry/CMSCommonData/data/cavernData/2017/v1/cavernData.xml',
        'Geometry/EcalCommonData/data/PhaseII/v2/eregalgo.xml',
        'Geometry/EcalCommonData/data/PhaseII/v2/ectkcable.xml',
        'Geometry/EcalCommonData/data/PhaseII/v2/ectkcablemat.xml',
        'Geometry/HcalCommonData/data/hcalrotations.xml',
        'Geometry/HcalCommonData/data/hcal/v2/hcalalgo.xml',
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml',
        'Geometry/HcalCommonData/data/hcalcablealgo/v2/hcalcablealgo.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcal/v10/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalEE/v10/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalHEsil/v10/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/hgcalHEmix/v10/hgcalHEmix.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v9/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v9/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalCons/v10/hgcalCons.xml',
    )+
    cms.vstring(
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)
