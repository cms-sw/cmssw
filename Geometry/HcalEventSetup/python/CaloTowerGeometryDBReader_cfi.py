import FWCore.ParameterSet.Config as cms

CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
                                            applyAlignment = cms.bool(False)
                                            )
