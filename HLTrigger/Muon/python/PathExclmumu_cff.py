# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

#L2 muon isolation

#L3 muon

#L3 muon isolation

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
# HLT Filter flux ##########################################################
ExclMMLevel1Seed = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleExclMM = copy.deepcopy(hltPrescaler)
ExclMML1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("ExclMMLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

ExclMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("ExclMML1Filtered"),
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

ExclDiMuonIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("ExclMML2Filtered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

ExclMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(4.0),
    PreviousCandTag = cms.InputTag("ExclDiMuonIsoL2IsoFiltered"),
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

ExclDiMuonIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("ExclMML3Filtered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

diMuonExcl = cms.Sequence(prescaleExclMM+l1muonreco+ExclMMLevel1Seed+ExclMML1Filtered+l2muonreco+ExclMML2Filtered+l2muonisoreco+ExclDiMuonIsoL2IsoFiltered+l3muonreco+ExclMML3Filtered+l3muonisoreco+ExclDiMuonIsoL3IsoFiltered)
ExclMMLevel1Seed.L1SeedsLogicalExpression = 'L1_DoubleMu3'

