import FWCore.ParameterSet.Config as cms

tpparams12 = cms.ESSource("EmptyESSource", 
                      recordName = cms.string("EcalTPGPhysicsConstRcd"),
                      firstValid = cms.vuint32(1),
                      iovIsRunNotTime = cms.bool(True)
                     )
