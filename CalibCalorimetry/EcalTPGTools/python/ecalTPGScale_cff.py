import FWCore.ParameterSet.Config as cms

# ESProducer for EcalTPGPhysicsConst
# esmodule creating  records + corresponding empty essource
from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff import *

tpparams12 = cms.ESSource("EmptyESSource", 
                      recordName = cms.string("EcalTPGPhysicsConstRcd"),
                      firstValid = cms.vuint32(1),
                      iovIsRunNotTime = cms.bool(True)
                     )
