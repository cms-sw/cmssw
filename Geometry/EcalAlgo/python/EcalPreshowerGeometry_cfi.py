import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
                                         applyAlignment = cms.bool(False),
                                         appendToDataLabel = cms.string("_master")
                                         )

EcalPreshowerGeometryFromDBEP = cms.ESProducer("EcalPreshowerGeometryFromDBEP",
                                               applyAlignment = cms.bool(False)
                                               )

EcalPreshowerGeometryToDBEP = cms.ESProducer("EcalPreshowerGeometryToDBEP",
                                          applyAlignment = cms.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
                                          )
