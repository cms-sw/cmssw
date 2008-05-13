# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.PathDiMuon_1032_NoIso_cff import *
import HLTrigger.HLTcore.hltPrescaler_cfi
# HLT Filter flux ##########################################################
hltPrescalehltDiMuonNoIsoRelaxedVtx2mm = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    #int32 MinNhits = 5
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MaxDr = cms.double(0.2),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

diMuonNoIsoRelaxedVtx2mm = cms.Sequence(hltPrescalehltDiMuonNoIsoRelaxedVtx2mm+hltL1muonrecoSequence+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+hltL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+hltL3muonrecoSequence+hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm)

