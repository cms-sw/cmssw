import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer( "EcalBarrelGeometryEP",
                                       applyAlignment = cms.bool(False),
                                       appendToDataLabel = cms.string("_master")
                                       )

EcalBarrelGeometryToDBEP = cms.ESProducer( "EcalBarrelGeometryToDBEP",
                                           applyAlignment = cms.bool(False),
                                           appendToDataLabel = cms.string("_toDB")
                                           )

