import FWCore.ParameterSet.Config as cms
from DQM.GEM.gemEfficiencyAnalyzer_cfi import *

gemEfficiencyAnalyzerCosmics = gemEfficiencyAnalyzer.clone(
    isCosmics = True,
)

gemEfficiencyAnalyzerCosmicsTwoLeg = gemEfficiencyAnalyzerCosmics.clone(
    muonTag = 'muons',
    name = 'Cosmic 2-Leg STA Muon',
    folder = 'GEM/Efficiency/type1'
)

gemEfficiencyAnalyzerCosmicsOneLeg = gemEfficiencyAnalyzerCosmics.clone(
    muonTag = 'muons1Leg',
    name = 'Cosmic 1-Leg STA Muon',
    folder = 'GEM/Efficiency/type2'
)
