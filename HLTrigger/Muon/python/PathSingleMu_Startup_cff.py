# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_cff import *
import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
# HLT Filter flux ##########################################################
SingleMuStartupLevel1Seed = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleSingleMuStartup = copy.deepcopy(hltPrescaler)
SingleMuStartupL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("SingleMuStartupLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

SingleMuStartupL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("SingleMuStartupL1Filtered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    #int32 MinNhits = 4
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("offlineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

singleMuStartup = cms.Sequence(prescaleSingleMuStartup+l1muonreco+SingleMuStartupLevel1Seed+SingleMuStartupL1Filtered+l2muonreco+SingleMuStartupL2PreFiltered)
SingleMuStartupLevel1Seed.L1SeedsLogicalExpression = 'L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7'

