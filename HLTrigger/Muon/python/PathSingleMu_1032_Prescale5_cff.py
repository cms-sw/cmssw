# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_1032_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltSingleMuPrescale5Level1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescaleSingleMuPrescale5 = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltSingleMuPrescale5L1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuPrescale5Level1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuPrescale5L2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale5L1Filtered"),
    MinPt = cms.double(5.0),
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

hltSingleMuPrescale5L3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale5L2PreFiltered"),
    MinPt = cms.double(5.0),
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

singleMuPrescale5 = cms.Sequence(hltPrescaleSingleMuPrescale5+hltL1muonrecoSequence+hltSingleMuPrescale5Level1Seed+hltSingleMuPrescale5L1Filtered+hltL2muonrecoSequence+hltSingleMuPrescale5L2PreFiltered+hltL3muonrecoSequence+hltSingleMuPrescale5L3PreFiltered)
hltSingleMuPrescale5Level1Seed.L1SeedsLogicalExpression = 'L1_SingleMu5'

