import FWCore.ParameterSet.Config as cms

EcalShashlikGeometryEP = cms.ESProducer( "EcalShashlikGeometryEP",
                                       applyAlignment = cms.bool(False),
                                       appendToDataLabel = cms.string("_master")
                                       )

EcalShashlikGeometryToDBEP = cms.ESProducer( "EcalShashlikGeometryToDBEP",
                                           applyAlignment = cms.bool(False),
                                           appendToDataLabel = cms.string("_toDB")
                                           )
