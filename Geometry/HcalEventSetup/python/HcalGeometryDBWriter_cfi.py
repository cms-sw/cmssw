import FWCore.ParameterSet.Config as cms

HcalHardcodeGeometryEP = cms.ESProducer( "HcalHardcodeGeometryEP" ,
                                         appendToDataLabel = cms.string("_master"),
                                         UseOldLoader = cms.bool(False)
                                         )

HcalGeometryToDBEP = cms.ESProducer( "HcalGeometryToDBEP" ,
                                     applyAlignment = cms.bool(False) ,
                                     appendToDataLabel = cms.string("_toDB")
                                     )

