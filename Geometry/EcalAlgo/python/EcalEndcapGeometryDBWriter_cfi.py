import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryEP = cms.ESProducer( "EcalEndcapGeometryEP",
                                       applyAlignment = cms.bool(False),
                                       appendToDataLabel = cms.string("_master")
                                       )

EcalEndcapGeometryToDBEP = cms.ESProducer( "EcalEndcapGeometryToDBEP",
                                           applyAlignment = cms.bool(False),
                                           appendToDataLabel = cms.string("_toDB")
                                           )
