import FWCore.ParameterSet.Config as cms

HcalHardcodeGeometryEP = cms.ESProducer( "HcalHardcodeGeometryEP" ,
                                         appendToDataLabel = cms.string("_master")
                                         )

HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
                                       applyAlignment = cms.bool(False)
                                       )

HcalGeometryToDBEP = cms.ESProducer( "HcalGeometryToDBEP" ,
                                     applyAlignment = cms.bool(False) ,
                                     appendToDataLabel = cms.string("_toDB")
                                     )

