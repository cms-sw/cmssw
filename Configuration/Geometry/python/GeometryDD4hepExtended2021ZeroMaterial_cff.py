import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                    confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021ZeroMaterial.xml'),
                                    appendToDataLabel = cms.string('')
)

DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                             appendToDataLabel = cms.string('')
)

DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                            appendToDataLabel = cms.string(''))

DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                         appendToDataLabel = cms.string('')
)

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *
from Geometry.MuonNumbering.muonOffsetESProducer_cff import *
