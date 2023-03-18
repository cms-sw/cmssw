import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
        'Geometry/CMSCommonData/data/materials/2021/v1/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/CMSCommonData/data/extend/v2/cmsextent.xml',
        'Geometry/CMSCommonData/data/cavernData/2021/v1/cavernData.xml',
        'Geometry/CMSCommonData/data/cms/2026/v5/cms.xml',
        'Geometry/CMSCommonData/data/cmsMother.xml',
        'Geometry/CMSCommonData/data/eta3/etaMax.xml',
        'Geometry/CMSCommonData/data/muonBase/2026/v5/muonBase.xml',
        'Geometry/CMSCommonData/data/cmsMuon.xml',
        'Geometry/CMSCommonData/data/mgnt.xml',
        'Geometry/CMSCommonData/data/muonMB.xml',
        'Geometry/CMSCommonData/data/muonMagnet.xml',
        'Geometry/MuonCommonData/data/muonYoke/2026/v3/muonYoke.xml',
        'Geometry/MuonCommonData/data/mf/2026/v8/mf.xml',
        'Geometry/MuonCommonData/data/gemf/TDR_BaseLine/gemf.xml',
        'Geometry/MuonCommonData/data/gem11/TDR_BaseLine/gem11.xml',
        'Geometry/MuonCommonData/data/gem21/TDR_Eta16/gem21.xml',
        'Geometry/MuonCommonData/data/mfshield/2026/v6/mfshield.xml',
        'Geometry/MuonCommonData/data/ge0/TDR_Dev/v4/ge0.xml',
        'Geometry/MuonCommonData/data/ge0shield/2026/v1/ge0shield.xml',
        'Geometry/CMSCommonData/data/FieldParameters.xml',
    ),
    rootNodeName = cms.string('cms:OCMS')
)
