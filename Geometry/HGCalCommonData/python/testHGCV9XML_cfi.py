
import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/cms/2019/v1/cms.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/caloBase/2026/v1/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/data/hgcal/v9/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalEE/v9/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalHEsil/v9/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/hgcalHEmix/v9/hgcalHEmix.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v9/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v9/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalCons/v9/hgcalCons.xml',
        'Geometry/HGCalSimData/data/hgcsensv9.xml',
        'Geometry/HGCalSimData/data/hgcProdCutsv9.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)


