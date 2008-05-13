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
hltSingleMuIsoLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescaleSingleMuIso = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltSingleMuIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuIsoL2PreFiltered = cms.EDFilter("HLTMuonPreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL1Filtered"),
    MinPt = cms.double(19.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltSingleMuIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL2IsoFiltered"),
    MinPt = cms.double(19.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltSingleMuIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

singleMuIso = cms.Sequence(hltPrescaleSingleMuIso+hltL1muonrecoSequence+hltSingleMuIsoLevel1Seed+hltSingleMuIsoL1Filtered+hltL2muonrecoSequence+hltSingleMuIsoL2PreFiltered+hltL2muonisorecoSequence+hltSingleMuIsoL2IsoFiltered+hltL3muonrecoSequence+hltSingleMuIsoL3PreFiltered+hltL3muonisorecoSequence+hltSingleMuIsoL3IsoFiltered)
hltSingleMuIsoLevel1Seed.L1SeedsLogicalExpression = 'L1_SingleMu14'

