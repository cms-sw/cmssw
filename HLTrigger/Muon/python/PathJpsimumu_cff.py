# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L3 muon

import FWCore.ParameterSet.Config as cms

from HLTrigger.Muon.CommonModules_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
hltJpsiMMLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescaleJpsiMM = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltJpsiMML1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltJpsiMMLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltJpsiMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltJpsiMML1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(5.0),
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
    MinInvMass = cms.double(1.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltJpsiMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltJpsiMML2Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(3.9),
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
    MinInvMass = cms.double(3.5),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

jpsiMM = cms.Sequence(hltPrescaleJpsiMM+hltL1muonrecoSequence+hltJpsiMMLevel1Seed+hltJpsiMML1Filtered+hltL2muonrecoSequence+hltJpsiMML2Filtered+hltL3muonrecoSequence+hltJpsiMML3Filtered)
hltJpsiMMLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

