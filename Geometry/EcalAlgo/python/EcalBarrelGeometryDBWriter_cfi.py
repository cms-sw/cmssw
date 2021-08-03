import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
                                      applyAlignment = cms.bool(False),
                                      appendToDataLabel = cms.string("_master")
)

_EcalBarrelGeometryEP_dd4hep = cms.ESProducer("EcalBarrelGeometryEPdd4hep",
                                              applyAlignment = cms.bool(False),
                                              appendToDataLabel = cms.string("_master")
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(EcalBarrelGeometryEP, _EcalBarrelGeometryEP_dd4hep)

EcalBarrelGeometryToDBEP = cms.ESProducer("EcalBarrelGeometryToDBEP",
                                          applyAlignment = cms.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
)
