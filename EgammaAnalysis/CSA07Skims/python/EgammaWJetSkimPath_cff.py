import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaWJetToEleHLT_cfi import *
from EgammaAnalysis.CSA07Skims.EgammaWJetToMuHLT_cfi import *
from EgammaAnalysis.CSA07Skims.EgammaWJetToElePlusProbe_cfi import *
from EgammaAnalysis.CSA07Skims.EgammaWJetToMuPlusProbe_cfi import *
electronFilterWPath = cms.Path(EgammaWJetToEleHLT+EgammaWJetToElePlusProbe)
muonFilterWPath = cms.Path(EgammaWJetToMuHLT+EgammaWJetToMuPlusProbe)

