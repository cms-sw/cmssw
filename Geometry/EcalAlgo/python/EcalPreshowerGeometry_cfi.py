import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
                                         applyAlignment = cms.untracked.bool(False),
                                         appendToDataLabel = cms.string("_master")
                                         )

EcalPreshowerGeometryFromDBEP = cms.ESProducer("EcalPreshowerGeometryFromDBEP",
                                               applyAlignment = cms.untracked.bool(False)
                                               )

EcalPreshowerGeometryToDBEP = cms.ESProducer("EcalPreshowerGeometryToDBEP",
                                          applyAlignment = cms.untracked.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
                                          )
