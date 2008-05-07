import FWCore.ParameterSet.Config as cms

tpparams12 = ESSource("EmptyESSource", 
                      recordName = cms.string("EcalTPGPhysicsConstRcd"),
                      firstValid = cms.vuint32( ),
                      bool iovIsRunNotTime = cms.bool(True)
                     )
