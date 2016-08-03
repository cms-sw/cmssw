import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/cmsextent.xml',
        'Geometry/CMSCommonData/data/PhaseI/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/PhaseII/caloBase.xml',
        'Geometry/CMSCommonData/data/cmsCalo.xml',
        'Geometry/HcalCommonData/data/hcalrotations.xml',
        'Geometry/HcalCommonData/data/PhaseII/HGCal/hcalalgo.xml',
        'Geometry/HcalCommonData/data/PhaseII/HGCal/hcalendcapalgo.xml',
        'Geometry/HcalCommonData/data/PhaseII/hcalSimNumbering.xml',
        'Geometry/HcalCommonData/data/PhaseII/hcalRecNumberingRebuild.xml',
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml',
        'Geometry/HcalCommonData/data/PhaseII/HGCal/hcalsenspmf.xml',
        'Geometry/HGCalCommonData/data/v7/hgcal.xml',
        'Geometry/HGCalCommonData/data/v7/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/v7/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/v7/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/v7/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


