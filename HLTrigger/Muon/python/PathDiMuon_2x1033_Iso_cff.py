# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L2 muon isolation

#L3 muon

#L3 muon isolation

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_2x1033_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltDiMuonIsoLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescalehltDiMuonIso = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltDiMuonIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltDiMuonIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltDiMuonIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonIsoL1Filtered"),
    MinPt = cms.double(7.0),
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

hltDiMuonIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltDiMuonIsoL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltDiMuonIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonIsoL2IsoFiltered"),
    MinPt = cms.double(7.0),
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

hltDiMuonIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltDiMuonIsoL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

diMuonIso = cms.Sequence(hltPrescalehltDiMuonIso+hltL1muonrecoSequence+hltDiMuonIsoLevel1Seed+hltDiMuonIsoL1Filtered+hltL2muonrecoSequence+hltDiMuonIsoL2PreFiltered+hltL2muonisorecoSequence+hltDiMuonIsoL2IsoFiltered+hltL3muonrecoSequence+hltDiMuonIsoL3PreFiltered+hltL3muonisorecoSequence+hltDiMuonIsoL3IsoFiltered)
hltDiMuonIsoLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

