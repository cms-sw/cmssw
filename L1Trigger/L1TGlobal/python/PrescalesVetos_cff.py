#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

L1TGlobalPrescalesVetosRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TGlobalPrescalesVetosRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

L1TGlobalPrescalesVetos = cms.ESProducer("L1TGlobalPrescalesVetosESProducer",

    # dummy version, no parameters yet...                                        
 
)


