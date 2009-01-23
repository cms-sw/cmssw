import FWCore.ParameterSet.Config as cms

CastorHardcodeGeometryEP = cms.ESProducer("CastorHardcodeGeometryEP",
                                          applyAlignment = cms.untracked.bool(False),
                                          appendToDataLabel = cms.string("_master")
                                          )


CastorGeometryFromDBEP = cms.ESProducer( "CastorGeometryFromDBEP",
                                         applyAlignment = cms.untracked.bool(False)
                                         )

CastorGeometryToDBEP = cms.ESProducer( "CastorGeometryToDBEP" ,
                                       applyAlignment = cms.untracked.bool(False) ,
                                       appendToDataLabel = cms.string("_toDB")
                                       )

