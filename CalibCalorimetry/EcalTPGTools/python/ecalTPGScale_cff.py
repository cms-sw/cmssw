import FWCore.ParameterSet.Config as cms

tpparams12 = ESSource("EmptyESSource", 
                      recordName = cms.string("EcalTPGPhysicsConstRcd"),
                      firstValid = cms.vuint32(1),
                      iovIsRunNotTime = cms.bool(True)
                     )
