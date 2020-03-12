import FWCore.ParameterSet.Config as cms

from Geometry.EcalCommonData.ecalSimulationParametersEB_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(ecalSimulationParametersEB,
                fromDD4Hep = cms.bool(True)
)

ecalSimulationParametersEE = ecalSimulationParametersEB.clone(
    name  = cms.string("EcalHitsEE"))

ecalSimulationParametersES = ecalSimulationParametersEB.clone(
    name  = cms.string("EcalHitsES"))
