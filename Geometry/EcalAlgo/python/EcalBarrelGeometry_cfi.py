import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
                                          applyAlignment = cms.untracked.bool(False)
                                      )

EcalBarrelGeometryFromDBEP = cms.ESProducer("EcalBarrelGeometryFromDBEP",
                                          applyAlignment = cms.untracked.bool(False)
                                      )
