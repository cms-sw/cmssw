import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
                                      applyAlignment = cms.untracked.bool(False),
                                      appendToDataLabel = cms.string("_master")
                                      )

EcalBarrelGeometryFromDBEP = cms.ESProducer("EcalBarrelGeometryFromDBEP",
                                            applyAlignment = cms.untracked.bool(False)
                                            )

EcalBarrelGeometryToDBEP = cms.ESProducer("EcalBarrelGeometryToDBEP",
                                          applyAlignment = cms.untracked.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
                                          )

