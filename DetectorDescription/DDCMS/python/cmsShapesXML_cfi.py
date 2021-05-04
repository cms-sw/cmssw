import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring("DetectorDescription/DDCMS/data/cmsWorld.xml",
                               "DetectorDescription/DDCMS/data/testMaterials.xml",
                               "DetectorDescription/DDCMS/data/testLogicalParts.xml",
                               "DetectorDescription/DDCMS/data/testRotations.xml",
                               "DetectorDescription/DDCMS/data/testSolids.xml",
                               "DetectorDescription/DDCMS/data/testPosParts.xml",
                               "DetectorDescription/DDCMS/data/testAlgorithm.xml",
                               "Geometry/CMSCommonData/data/materials/2015/v1/materials.xml"),
    rootNodeName = cms.string('cmsWorld:OCMS')
)
