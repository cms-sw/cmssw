import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryFromDBEP = cms.ESProducer( "EcalBarrelGeometryFromDBEP",
                                             applyAlignment = cms.bool(False)
                                             )

