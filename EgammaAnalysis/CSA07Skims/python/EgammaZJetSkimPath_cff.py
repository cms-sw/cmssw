import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.CSA07Skims.EgammaZJetToEleHLT_cfi import *
from EgammaAnalysis.CSA07Skims.EgammaZJetToMuHLT_cfi import *
from EgammaAnalysis.CSA07Skims.EgammaZJetToElePlusProbe_cfi import *
from EgammaAnalysis.CSA07Skims.EgammaZJetToMuPlusProbe_cfi import *
electronFilterZPath = cms.Path(EgammaZJetToEleHLT+EgammaZJetToElePlusProbe)
muonFilterZPath = cms.Path(EgammaZJetToMuHLT+EgammaZJetToMuPlusProbe)

