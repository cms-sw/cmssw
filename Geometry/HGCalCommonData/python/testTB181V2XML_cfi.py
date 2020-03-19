import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
                               'Geometry/HGCalCommonData/data/TB181/cms.xml',
                               'Geometry/HGCalCommonData/data/TB181/Oct181/hgcal.xml',
                               'Geometry/HGCalCommonData/data/TB181/Test/hgcalHE.xml',
                               'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
                               'Geometry/HGCalCommonData/data/TB181/Test/hgcalsense.xml',
                               'Geometry/HGCalCommonData/data/TB181/Test/hgcProdCuts.xml',
                               'Geometry/HGCalCommonData/data/TB181/Test/hgcalCons.xml'
                               ),
    rootNodeName = cms.string('cms:OCMS')
)
