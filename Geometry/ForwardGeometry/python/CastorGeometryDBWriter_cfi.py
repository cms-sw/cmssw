import FWCore.ParameterSet.Config as cms

CastorHardcodeGeometryEP = cms.ESProducer("CastorHardcodeGeometryEP",
                                          appendToDataLabel = cms.string("_master")
                                          )

CastorGeometryToDBEP = cms.ESProducer( "CastorGeometryToDBEP" ,
                                       applyAlignment = cms.bool(False) ,
                                       appendToDataLabel = cms.string("_toDB")
                                       )

