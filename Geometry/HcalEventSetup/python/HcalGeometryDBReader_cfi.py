import FWCore.ParameterSet.Config as cms

HcalGeometryFromDBEP = cms.ESProducer("HcalGeometryFromDBEP",
                                      applyAlignment = cms.bool(False)
                                      )

HcalAlignmentEP = cms.ESProducer("HcalAlignmentEP")

