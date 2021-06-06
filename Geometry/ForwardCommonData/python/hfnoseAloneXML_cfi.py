import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2017/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/cms/2026/v5/cms.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/caloBase/2026/v5/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/ForwardCommonData/data/hfnose/v4/hfnose.xml',
        'Geometry/ForwardCommonData/data/hfnoseCell/v1/hfnoseCell.xml',
        'Geometry/ForwardCommonData/data/hfnoseWafer/v1/hfnoseWafer.xml',
        'Geometry/ForwardCommonData/data/hfnoseLayer/v2/hfnoseLayer.xml'),
    rootNodeName = cms.string('cms:OCMS')
)
