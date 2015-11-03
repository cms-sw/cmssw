import FWCore.ParameterSet.Config as cms

HGCalEEGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
                                           ##appendToDataLabel = cms.string("_master"),
                                           Name = cms.untracked.string("HGCalEESensitive"),
                                           applyAlignment = cms.bool(False)
                                           )

HGCalHESilGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
                                              appendToDataLabel = cms.string("_master"),
                                              Name = cms.untracked.string("HGCalHESiliconSensitive"),
                                              applyAlignment = cms.bool(False)
                                              )

HGCalHESciGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
                                              appendToDataLabel = cms.string("_master"),
                                              Name = cms.untracked.string("HGCalHEScintillatorSensitive"),
                                              applyAlignment = cms.bool(False)
                                              )

HGCalGeometryToDBEP = cms.ESProducer("HGCalGeometryToDBEP" ,
                                     applyAlignment = cms.bool(False) ,
                                     appendToDataLabel = cms.string("_toDB")
                                     )
