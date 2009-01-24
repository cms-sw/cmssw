import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
                                      applyAlignment = cms.bool(False),
                                      appendToDataLabel = cms.string("_master")
                                      )

EcalBarrelGeometryFromDBEP = cms.ESProducer("EcalBarrelGeometryFromDBEP",
                                            applyAlignment = cms.bool(False)
                                            )

EcalBarrelGeometryToDBEP = cms.ESProducer("EcalBarrelGeometryToDBEP",
                                          applyAlignment = cms.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
                                          )

