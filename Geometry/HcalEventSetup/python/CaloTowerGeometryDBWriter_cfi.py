import FWCore.ParameterSet.Config as cms

CaloTowerHardcodeGeometryEP = cms.ESProducer( "CaloTowerHardcodeGeometryEP" ,
                                              appendToDataLabel = cms.string("_master")
                                              )

CaloTowerGeometryToDBEP = cms.ESProducer( "CaloTowerGeometryToDBEP" ,
                                          applyAlignment = cms.bool(False) ,
                                          appendToDataLabel = cms.string("_toDB")
                                          )

