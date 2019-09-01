import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring("DetectorDescription/DDCMS/data/cmsWorld.xml",
                               "DetectorDescription/DDCMS/data/testMaterials.xml",
                               "DetectorDescription/DDCMS/data/testPseudoTrapSolids.xml",
                               "DetectorDescription/DDCMS/data/testRotations.xml",
                               "DetectorDescription/DDCMS/data/testLogicalParts.xml",
                               "DetectorDescription/DDCMS/data/materials.xml"),
    rootNodeName = cms.string('cmsWorld:OCMS')
)
