import FWCore.ParameterSet.Config as cms

from HLTrigger.Muon.CommonModules_1032_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
SingleMuPrescale710Level1Seed = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
preSingleMuPrescale710 = copy.deepcopy(hltPrescaler)
SingleMuPrescale710L1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("SingleMuPrescale710Level1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

SingleMuPrescale710L2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("SingleMuPrescale710L1Filtered"),
    MinPt = cms.double(10.0),
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

SingleMuPrescale710L3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("SingleMuPrescale710L2PreFiltered"),
    MinPt = cms.double(10.0),
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

singleMuPrescale710 = cms.Sequence(preSingleMuPrescale710+l1muonreco+SingleMuPrescale710Level1Seed+SingleMuPrescale710L1Filtered+l2muonreco+SingleMuPrescale710L2PreFiltered+l3muonreco+SingleMuPrescale710L3PreFiltered)
SingleMuPrescale710Level1Seed.L1SeedsLogicalExpression = 'L1_SingleMu7'
preSingleMuPrescale710.prescaleFactor = 100

