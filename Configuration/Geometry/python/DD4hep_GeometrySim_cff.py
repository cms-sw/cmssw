import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                    confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                    appendToDataLabel = cms.string('')
)

DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                             appendToDataLabel = cms.string('')
)
