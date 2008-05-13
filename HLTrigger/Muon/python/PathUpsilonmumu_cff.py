# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltUpsilonMMLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescalehltUpsilonMM = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltUpsilonMML1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltUpsilonMMLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltUpsilonMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltUpsilonMML1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(13.0),
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
    MinInvMass = cms.double(6.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltUpsilonMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltUpsilonMML2Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(11.0),
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
    MinInvMass = cms.double(8.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

upsilonMM = cms.Sequence(hltPrescalehltUpsilonMM+hltL1muonrecoSequence+hltUpsilonMMLevel1Seed+hltUpsilonMML1Filtered+hltL2muonrecoSequence+hltUpsilonMML2Filtered+hltL3muonrecoSequence+hltUpsilonMML3Filtered)
hltUpsilonMMLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

