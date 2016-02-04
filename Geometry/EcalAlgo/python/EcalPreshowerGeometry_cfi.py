import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryEP = cms.ESProducer( "EcalPreshowerGeometryEP",
                                          applyAlignment = cms.bool(False)
                                          )
