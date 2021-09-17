import FWCore.ParameterSet.Config as cms

from Geometry.EcalCommonData.ecalSimulationParametersAnalyzerEB_cfi import *

ecalSimulationParametersAnalyzerEE = ecalSimulationParametersAnalyzerEB.clone(
    name  = "EcalHitsEE")

ecalSimulationParametersAnalyzerES = ecalSimulationParametersAnalyzerEB.clone(
    name  = "EcalHitsES")
