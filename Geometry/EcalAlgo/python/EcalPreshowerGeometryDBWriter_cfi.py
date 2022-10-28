import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
                                         applyAlignment = cms.bool(False),
                                         appendToDataLabel = cms.string("_master")
)

_EcalPreshowerGeometryEP_dd4hep = cms.ESProducer("EcalPreshowerGeometryEPdd4hep",
                                                 applyAlignment = cms.bool(False),
                                                 appendToDataLabel = cms.string("_master")
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(EcalPreshowerGeometryEP, _EcalPreshowerGeometryEP_dd4hep)

EcalPreshowerGeometryToDBEP = cms.ESProducer("EcalPreshowerGeometryToDBEP",
                                             applyAlignment = cms.bool(False),
                                             appendToDataLabel = cms.string("_toDB")
)
