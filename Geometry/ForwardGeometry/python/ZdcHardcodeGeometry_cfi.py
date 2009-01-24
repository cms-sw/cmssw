import FWCore.ParameterSet.Config as cms

ZdcHardcodeGeometryEP = cms.ESProducer( "ZdcHardcodeGeometryEP",
                                        appendToDataLabel = cms.string("_master")
                                        )

ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
                                      applyAlignment = cms.bool(False)
                                      )

ZdcGeometryToDBEP = cms.ESProducer( "ZdcGeometryToDBEP" ,
                                    applyAlignment = cms.bool(False) ,
                                    appendToDataLabel = cms.string("_toDB")
                                    )
