import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms/2017/v1/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/caloBase/2023/v1/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HGCalCommonData/data/hgcal/v5/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalEE/v5/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalsil/v5/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/hgcalsci/v5/hgcalHEsci.xml',
        'Geometry/HGCalCommonData/data/hgcalCons/v5/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


