import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryEP = cms.ESProducer( "EcalPreshowerGeometryEP",
                                          applyAlignment = cms.bool(False),
                                          appendToDataLabel = cms.string("_master")
                                          )

EcalPreshowerGeometryToDBEP = cms.ESProducer( "EcalPreshowerGeometryToDBEP",
                                              applyAlignment = cms.bool(False),
                                              appendToDataLabel = cms.string("_toDB")
                                              )
