import FWCore.ParameterSet.Config as cms

ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
                                      applyAlignment = cms.bool(False)
                                      )
