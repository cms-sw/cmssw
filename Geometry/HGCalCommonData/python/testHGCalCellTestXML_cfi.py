import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/vtest/hgcal.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/vtest/hgcalcell.xml',
        'Geometry/HGCalCommonData/data/hgcalcell/vtest/hgcalpos.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


