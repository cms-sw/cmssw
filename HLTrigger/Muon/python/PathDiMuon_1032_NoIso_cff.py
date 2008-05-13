# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_1032_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltDiMuonNoIsoLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescalehltDiMuonNoIso = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltDiMuonNoIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltDiMuonNoIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltDiMuonNoIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL1Filtered"),
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

hltDiMuonNoIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL2PreFiltered"),
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

diMuonNoIso = cms.Sequence(hltPrescalehltDiMuonNoIso+hltL1muonrecoSequence+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+hltL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+hltL3muonrecoSequence+hltDiMuonNoIsoL3PreFiltered)
hltDiMuonNoIsoLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

