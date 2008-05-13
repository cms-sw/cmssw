# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.PathSingleMu_1032_NoIso_cff import *
import HLTrigger.HLTcore.hltPrescaler_cfi
# HLT Filter flux ##########################################################
hltPrescaleSingleMuNoIsoRelaxedVtx2cm = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(16.0),
    MinN = cms.int32(1),
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

singleMuNoIsoRelaxedVtx2cm = cms.Sequence(hltPrescaleSingleMuNoIsoRelaxedVtx2cm+hltL1muonrecoSequence+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+hltL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered+hltL3muonrecoSequence+hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm)

