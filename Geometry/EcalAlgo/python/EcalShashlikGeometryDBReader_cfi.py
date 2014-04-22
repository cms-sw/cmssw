import FWCore.ParameterSet.Config as cms

EcalShashlikGeometryFromDBEP = cms.ESProducer( "EcalShashlikGeometryFromDBEP",
                                             applyAlignment = cms.bool(False)
                                             )
