import FWCore.ParameterSet.Config as cms
from DQM.GEM.gemEfficiencyAnalyzer_cfi import *

################################################################################
# Tight global muons
################################################################################

gemDQMTightGlbMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string(
        "isGlobalMuon"
        "&& globalTrack.isNonnull"
        "&& passed('CutBasedIdTight')"
    ),
    filter = cms.bool(False)
)

gemEfficiencyAnalyzerTightGlb = gemEfficiencyAnalyzer.clone(
    muonTag = "gemDQMTightGlbMuons",
    muonTrackType = "CombinedTrack",
    startingStateType = "OutermostMeasurementState",
    folder = "GEM/Efficiency/muonGLB",
    muonName = "Tight GLB Muon",
    propagationErrorRCut = 0.5, # cm
    propagationErrorPhiCut = 0.1, # degree
)

gemEfficiencyAnalyzerTightGlbSeq = cms.Sequence(
    cms.ignore(gemDQMTightGlbMuons) *
    gemEfficiencyAnalyzerTightGlb)

################################################################################
# Standalone muons
################################################################################
gemDQMStaMuons = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string(
        "isStandAloneMuon"
        "&& outerTrack.isNonnull"
    ),
    filter = cms.bool(False)
)

gemEfficiencyAnalyzerSta = gemEfficiencyAnalyzer.clone(
    muonTag = "gemDQMStaMuons",
    muonTrackType = "OuterTrack",
    startingStateType = "OutermostMeasurementState",
    folder = "GEM/Efficiency/muonSTA",
    muonName = "STA Muon",
    propagationErrorRCut = 0.5, # cm
    propagationErrorPhiCut = 0.2, # degree
)

gemEfficiencyAnalyzerStaSeq = cms.Sequence(
    cms.ignore(gemDQMStaMuons) *
    gemEfficiencyAnalyzerSta)
