import FWCore.ParameterSet.Config as cms

from Geometry.EcalCommonData.ecalSimulationParametersEB_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(ecalSimulationParametersEB,
                fromDD4hep = True
)

ecalSimulationParametersEE = ecalSimulationParametersEB.clone(
    name  = "EcalHitsEE")

ecalSimulationParametersES = ecalSimulationParametersEB.clone(
    name  = "EcalHitsES")
