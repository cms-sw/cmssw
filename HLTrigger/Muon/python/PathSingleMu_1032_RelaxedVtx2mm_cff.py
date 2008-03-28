# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.PathSingleMu_1032_NoIso_cff import *
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
# HLT Filter flux ##########################################################
prescaleSingleMuNoIsoRelaxedVtx2mm = copy.deepcopy(hltPrescaler)
SingleMuNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("SingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(16.0),
    MinN = cms.int32(1),
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

singleMuNoIsoRelaxedVtx2mm = cms.Sequence(prescaleSingleMuNoIsoRelaxedVtx2mm+l1muonreco+SingleMuNoIsoLevel1Seed+SingleMuNoIsoL1Filtered+l2muonreco+SingleMuNoIsoL2PreFiltered+l3muonreco+SingleMuNoIsoL3PreFilteredRelaxedVtx2mm)

