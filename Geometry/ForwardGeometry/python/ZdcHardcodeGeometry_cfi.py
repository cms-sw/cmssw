import FWCore.ParameterSet.Config as cms

ZdcHardcodeGeometryEP = cms.ESProducer( "ZdcHardcodeGeometryEP",
                                        applyAlignment = cms.untracked.bool(False),
                                        appendToDataLabel = cms.string("_master")
                                        )

ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
                                      applyAlignment = cms.untracked.bool(False)
                                      )

ZdcGeometryToDBEP = cms.ESProducer( "ZdcGeometryToDBEP" ,
                                    applyAlignment = cms.untracked.bool(False) ,
                                    appendToDataLabel = cms.string("_toDB")
                                    )
