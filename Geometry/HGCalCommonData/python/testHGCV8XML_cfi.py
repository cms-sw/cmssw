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
#        'Geometry/HcalCommonData/data/PhaseII/HGCal/hcalsenspmf.xml',
        'Geometry/HcalSimData/data/hf.xml',
        'Geometry/HcalSimData/data/hfpmt.xml',
        'Geometry/HcalSimData/data/hffibrebundle.xml',
        'Geometry/HcalSimData/data/CaloUtil.xml',
        'Geometry/HcalSimData/data/HcalProdCuts.xml',
        'Geometry/HGCalCommonData/data/v8/hgcal.xml',
        'Geometry/HGCalCommonData/data/v8/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/v8/hgcalHEsil.xml',
        'Geometry/HGCalCommonData/data/v7/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/v8/hgcalCons.xml',
        'Geometry/HGCalSimData/data/hgcsensv8.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


