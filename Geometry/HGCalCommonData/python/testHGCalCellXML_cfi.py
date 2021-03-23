import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v15/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v15/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v9/hgcalpos.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


