import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/hgcalMaterial/v3/hgcalMaterial.xml',
        'Geometry/HGCalCommonData/test/cms.xml',
        'Geometry/ForwardCommonData/data/hfnoseWafer/v2/hfnose.xml',
        'Geometry/ForwardCommonData/data/hfnoseCell/v2/hfnoseCell.xml',
        'Geometry/ForwardCommonData/data/hfnoseWafer/v2/hfnoseWafer.xml',
        'Geometry/ForwardCommonData/data/hfnoseWafer/v2/hfnosepos.xml'),
    rootNodeName = cms.string('cms:OCMS')
)



