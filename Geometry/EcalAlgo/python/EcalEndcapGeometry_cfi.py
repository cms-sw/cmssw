import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
                                      applyAlignment = cms.untracked.bool(False),
                                      appendToDataLabel = cms.string("xml")
                                      )

EcalEndcapGeometryFromDBEP = cms.ESProducer("EcalEndcapGeometryFromDBEP",
                                            applyAlignment = cms.untracked.bool(False)
                                            )
