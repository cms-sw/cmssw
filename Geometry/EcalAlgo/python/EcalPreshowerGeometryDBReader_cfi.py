import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryFromDBEP = cms.ESProducer( "EcalPreshowerGeometryFromDBEP",
                                                applyAlignment = cms.bool(False)
                                                )
