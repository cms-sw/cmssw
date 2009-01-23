import FWCore.ParameterSet.Config as cms

HcalHardcodeGeometryEP = cms.ESProducer( "HcalHardcodeGeometryEP" ,
                                         applyAlignment = cms.untracked.bool(False),
                                         appendToDataLabel = cms.string("_master")
                                         )

HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
                                       applyAlignment = cms.untracked.bool(False)
                                       )

HcalGeometryToDBEP = cms.ESProducer( "HcalGeometryToDBEP" ,
                                     applyAlignment = cms.untracked.bool(False) ,
                                     appendToDataLabel = cms.string("_toDB")
                                     )

