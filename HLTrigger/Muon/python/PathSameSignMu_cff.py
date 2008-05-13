# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L2 muon isolation
# & hltL2muonisorecoSequence & hltSameSignMuL2IsoFiltered
#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltSameSignhltMuLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescalehltSameSignMu = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltSameSignMuL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSameSignhltMuLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltSameSignMuL2PreFiltered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltSameSignMuL1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(1),
    MaxInvMass = cms.double(9999.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MinPtPair = cms.double(0.0),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltSameSignMuL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSameSignMuL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltSameSignMuL3PreFiltered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltSameSignMuL2PreFiltered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(1),
    MaxInvMass = cms.double(9999.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltSameSignMuL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSameSignMuL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

sameSignMu = cms.Sequence(hltPrescalehltSameSignMu+hltL1muonrecoSequence+hltSameSignhltMuLevel1Seed+hltSameSignMuL1Filtered+hltL2muonrecoSequence+hltSameSignMuL2PreFiltered+hltL3muonrecoSequence+hltSameSignMuL3PreFiltered)
hltSameSignhltMuLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

