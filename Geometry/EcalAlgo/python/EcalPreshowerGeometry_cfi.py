import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
                                                  applyAlignment = cms.untracked.bool(False)
                                              )

EcalPreshowerGeometryFromDBEP = cms.ESProducer("EcalPreshowerGeometryFromDBEP",
                                               applyAlignment = cms.untracked.bool(False)
                                               )
