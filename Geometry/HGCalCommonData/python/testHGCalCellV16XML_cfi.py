import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v16/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v16/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/v16/hgcalpos.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


