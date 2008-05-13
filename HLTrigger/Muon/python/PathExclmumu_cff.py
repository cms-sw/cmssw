# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L2 muon isolation

#L3 muon

#L3 muon isolation

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltExclMMLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescalehltExclMM = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltExclMML1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltExclMMLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltExclMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltExclMML1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(999999.0),
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
    MinInvMass = cms.double(-1.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltExclhltDiMuonIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltExclMML2Filtered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltExclMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(4.0),
    PreviousCandTag = cms.InputTag("hltExclhltDiMuonIsoL2IsoFiltered"),
    MaxPtBalance = cms.double(2.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(999999.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(4.0),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    MaxAcop = cms.double(3.2),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MinInvMass = cms.double(-1.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(3.0)
)

hltExclhltDiMuonIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltExclMML3Filtered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

diMuonhltExcl = cms.Sequence(hltPrescalehltExclMM+hltL1muonrecoSequence+hltExclMMLevel1Seed+hltExclMML1Filtered+hltL2muonrecoSequence+hltExclMML2Filtered+hltL2muonisorecoSequence+hltExclhltDiMuonIsoL2IsoFiltered+hltL3muonrecoSequence+hltExclMML3Filtered+hltL3muonisorecoSequence+hltExclhltDiMuonIsoL3IsoFiltered)
hltExclMMLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

