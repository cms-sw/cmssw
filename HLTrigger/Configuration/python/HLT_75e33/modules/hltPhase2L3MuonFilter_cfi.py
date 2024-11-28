import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonFilter = cms.EDProducer("phase2HLTMuonSelectorForL3",
    l1TkMuons = cms.InputTag("l1tTkMuonsGmt"),
    l2MuonsUpdVtx = cms.InputTag("hltL2MuonsFromL1TkMuon:UpdatedAtVtx"),
    l3Tracks = cms.InputTag("hltIter2Phase2L3FromL1TkMuonMerged"),
    IOFirst = cms.bool(True),
    matchingDr = cms.double(0.02),
    applyL3Filters = cms.bool(True),
    MinNhits = cms.int32(1),
    MaxNormalizedChi2 = cms.double(5.0),
    MinNhitsMuons = cms.int32(0),
    MinNhitsPixel = cms.int32(1),
    MinNhitsTracker = cms.int32(6),
    MaxPtDifference = cms.double(999.0),
)

from Configuration.ProcessModifiers.phase2L2AndL3Muons_cff import phase2L2AndL3Muons
from Configuration.ProcessModifiers.phase2L3MuonsOIFirst_cff import phase2L3MuonsOIFirst
(phase2L2AndL3Muons & phase2L3MuonsOIFirst).toModify(
    hltPhase2L3MuonFilter,
    l3Tracks = "hltPhase2L3OIMuonTrackSelectionHighPurity",
    IOFirst = False,
)
