# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.PathDiMuon_1032_NoIso_cff import *
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
# HLT Filter flux ##########################################################
prescaleDiMuonNoIsoRelaxedVtx2cm = copy.deepcopy(hltPrescaler)
DiMuonNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("DiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    #int32 MinNhits = 5
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

diMuonNoIsoRelaxedVtx2cm = cms.Sequence(prescaleDiMuonNoIsoRelaxedVtx2cm+l1muonreco+DiMuonNoIsoLevel1Seed+DiMuonNoIsoL1Filtered+l2muonreco+DiMuonNoIsoL2PreFiltered+l3muonreco+DiMuonNoIsoL3PreFilteredRelaxedVtx2cm)

