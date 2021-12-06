import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation

DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                             appendToDataLabel = cms.string('')
)

DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                            appendToDataLabel = cms.string(''))

DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                         appendToDataLabel = cms.string('')
)

from DetectorDescription.DDCMS.DDDetectorESProducerFromDB_cfi import *

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometryDB_cfi import *

from Geometry.EcalCommonData.ecalSimulationParameters_cff   import *
from Geometry.HcalCommonData.hcalSimDBConstants_cff         import *

from Geometry.MuonNumbering.muonGeometryConstants_cff       import *
from Geometry.MuonNumbering.muonOffsetESProducer_cff        import *
