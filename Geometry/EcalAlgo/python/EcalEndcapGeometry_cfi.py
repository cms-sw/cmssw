import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
                                      applyAlignment = cms.untracked.bool(False),
                                      appendToDataLabel = cms.string("_master")
                                      )

EcalEndcapGeometryFromDBEP = cms.ESProducer("EcalEndcapGeometryFromDBEP",
                                            applyAlignment = cms.untracked.bool(False)
                                            )

EcalEndcapGeometryToDBEP = cms.ESProducer("EcalEndcapGeometryToDBEP",
                                          applyAlignment = cms.untracked.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
                                          )
