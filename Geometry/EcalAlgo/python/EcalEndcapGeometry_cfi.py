import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
                                      applyAlignment = cms.bool(False),
                                      appendToDataLabel = cms.string("_master")
                                      )

EcalEndcapGeometryFromDBEP = cms.ESProducer("EcalEndcapGeometryFromDBEP",
                                            applyAlignment = cms.bool(False)
                                            )

EcalEndcapGeometryToDBEP = cms.ESProducer("EcalEndcapGeometryToDBEP",
                                          applyAlignment = cms.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
                                          )
