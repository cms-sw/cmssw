import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryEP = cms.ESProducer( "EcalEndcapGeometryEP",
                                       applyAlignment = cms.bool(False)
                                       )
