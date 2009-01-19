import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
                                      applyAlignment = cms.untracked.bool(False),
                                      appendToDataLabel = cms.string("xml")
                                      )

EcalBarrelGeometryFromDBEP = cms.ESProducer("EcalBarrelGeometryFromDBEP",
                                          applyAlignment = cms.untracked.bool(False)
                                      )
