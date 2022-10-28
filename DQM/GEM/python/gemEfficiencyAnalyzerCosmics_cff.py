import FWCore.ParameterSet.Config as cms
from DQM.GEM.gemEfficiencyAnalyzer_cfi import *

gemEfficiencyAnalyzerCosmics = gemEfficiencyAnalyzer.clone(
    scenario = "cosmics",
    propagationErrorRCut = 0.5, # cm
    propagationErrorPhiCut = 0.1, # degree
    muonPtMinCutGE11 = 0, # GeV
    muonEtaMinCutGE11 = 0.5,
    muonEtaMaxCutGE11 = 10.0,
    muonEtaNbinsGE11 = 30,
    muonEtaLowGE11 = 0.0,
    muonEtaUpGE11 = 3.0,
)

gemEfficiencyAnalyzerCosmicsGlb = gemEfficiencyAnalyzerCosmics.clone(
    muonTag = 'muons',
    muonTrackType = 'CombinedTrack',
    startingStateType = "OutermostMeasurementState",
    folder = 'GEM/Efficiency/muonGLB',
    muonName = 'Cosmic 2-Leg GLB Muon',
)

gemEfficiencyAnalyzerCosmicsSta = gemEfficiencyAnalyzerCosmics.clone(
    muonTag = 'muons',
    muonTrackType = 'OuterTrack',
    startingStateType = "OutermostMeasurementState",
    folder = 'GEM/Efficiency/muonSTA',
    muonName = 'Cosmic 2-Leg STA Muon',
)
