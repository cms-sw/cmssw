import FWCore.ParameterSet.Config as cms

EcalEndcapGeometryEP = cms.ESProducer( "EcalEndcapGeometryEP",
                                       applyAlignment = cms.bool(False)
                                       )

_EcalEndcapGeometryEP_dd4hep = cms.ESProducer("EcalEndcapGeometryEPdd4hep",
    applyAlignment = cms.bool(False)
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toReplaceWith(EcalEndcapGeometryEP, _EcalEndcapGeometryEP_dd4hep)
