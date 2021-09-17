import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
                                        geomXMLFiles = cms.vstring('DetectorDescription/Core/test/data/materials.xml',
                                                                   'DetectorDescription/Core/test/data/world.xml',
                                                                   'DetectorDescription/Core/test/data/assembly.xml'),
                                        rootNodeName = cms.string('world:MotherOfAllBoxes')
                                        )
