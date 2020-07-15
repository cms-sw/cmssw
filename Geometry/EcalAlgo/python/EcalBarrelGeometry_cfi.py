import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer( "EcalBarrelGeometryEP",
                                       applyAlignment = cms.bool(False)
                                      )

_EcalBarrelGeometryEP_dd4hep = cms.ESProducer("EcalBarrelGeometryEPdd4hep",
    applyAlignment = cms.bool(False)
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(EcalBarrelGeometryEP, _EcalBarrelGeometryEP_dd4hep)
