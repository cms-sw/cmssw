import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
                                      applyAlignment = cms.bool(False),
                                      appendToDataLabel = cms.string("_master")
)

_EcalEndcapGeometryEP_dd4hep = cms.ESProducer("EcalEndcapGeometryEPdd4hep",
                                              applyAlignment = cms.bool(False),
                                              appendToDataLabel = cms.string("_master")
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(EcalEndcapGeometryEP, _EcalEndcapGeometryEP_dd4hep)

EcalEndcapGeometryToDBEP = cms.ESProducer("EcalEndcapGeometryToDBEP",
                                          applyAlignment = cms.bool(False),
                                          appendToDataLabel = cms.string("_toDB")
)
