import FWCore.ParameterSet.Config as cms

HGcalGeometryFromDBEP = cms.ESProducer("HGcalGeometryFromDBEP",
                                       applyAlignment = cms.bool(False)
                                       )
