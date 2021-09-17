import FWCore.ParameterSet.Config as cms

EcalPreshowerGeometryEP = cms.ESProducer( "EcalPreshowerGeometryEP",
                                          applyAlignment = cms.bool(False)
                                          )

_EcalPreshowerGeometryEP_dd4hep = cms.ESProducer("EcalPreshowerGeometryEPdd4hep",
    applyAlignment = cms.bool(False)
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(EcalPreshowerGeometryEP, _EcalPreshowerGeometryEP_dd4hep)
