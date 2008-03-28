# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_1032_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
# HLT Filter flux ##########################################################
DiMuonNoIsoLevel1Seed = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleDiMuonNoIso = copy.deepcopy(hltPrescaler)
DiMuonNoIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("DiMuonNoIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

DiMuonNoIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("DiMuonNoIsoL1Filtered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

DiMuonNoIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("DiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

diMuonNoIso = cms.Sequence(prescaleDiMuonNoIso+l1muonreco+DiMuonNoIsoLevel1Seed+DiMuonNoIsoL1Filtered+l2muonreco+DiMuonNoIsoL2PreFiltered+l3muonreco+DiMuonNoIsoL3PreFiltered)
DiMuonNoIsoLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

