import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/MuonCommonData/data/cosmic1/cms.xml',
        #'Geometry/MuonCommonData/data/cosmic1/muonBase.xml', # Phase-2 Muon
        #'Geometry/MuonCommonData/data/cosmic1/cmsMuon.xml',        
        'Geometry/MuonCommonData/data/cosmic1/gembox.xml',
        #'Geometry/MuonCommonData/data/cosmic1/mf.xml',        
        'Geometry/MuonCommonData/data/v2/gemf.xml',
        'Geometry/MuonCommonData/data/cosmic1/gem11.xml',
        'Geometry/MuonCommonData/data/cosmic1/muonNumbering.xml',
        'Geometry/MuonCommonData/data/cosmic1/muonSens.xml',
        'Geometry/MuonCommonData/data/cosmic1/muonProdCuts.xml',
        'Geometry/MuonCommonData/data/cosmic1/GEMSpecsFilter.xml',   # Phase-2 Muon
        'Geometry/MuonCommonData/data/cosmic1/GEMSpecs.xml'),
    rootNodeName = cms.string('cms:OCMS')
)

