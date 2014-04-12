import FWCore.ParameterSet.Config as cms

CastorGeometryFromDBEP = cms.ESProducer( "CastorGeometryFromDBEP",
                                         applyAlignment = cms.bool(False)
                                         )

