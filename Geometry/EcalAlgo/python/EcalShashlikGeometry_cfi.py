import FWCore.ParameterSet.Config as cms

EcalShashlikGeometryEP = cms.ESProducer( "EcalShashlikGeometryEP",
                                         applyAlignment = cms.bool(False)
                                         )
