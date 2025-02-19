import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryFromDBEP = cms.ESProducer( "EcalEndcapGeometryFromDBEP",
                                             applyAlignment = cms.bool(False)
                                             )
