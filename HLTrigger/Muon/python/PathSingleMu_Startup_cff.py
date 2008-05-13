# The following comments couldn't be translated into the new config version:

#L1 muon

#L2 muon

import FWCore.ParameterSet.Config as cms

# RecoMuon flux ##########################################################
from HLTrigger.Muon.CommonModules_cff import *
import HLTrigger.HLTfilters.hltLevel1GTSeed_cfi
# HLT Filter flux ##########################################################
hltSingleMuStartupLevel1Seed = HLTrigger.HLTfilters.hltLevel1GTSeed_cfi.hltLevel1GTSeed.clone()
import HLTrigger.HLTcore.hltPrescaler_cfi
hltPrescaleSingleMuStartup = HLTrigger.HLTcore.hltPrescaler_cfi.hltPrescaler.clone()
hltSingleMuStartupL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuStartupLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuStartupL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuStartupL1Filtered"),
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

singleMuStartup = cms.Sequence(hltPrescaleSingleMuStartup+hltL1muonrecoSequence+hltSingleMuStartupLevel1Seed+hltSingleMuStartupL1Filtered+hltL2muonrecoSequence+hltSingleMuStartupL2PreFiltered)
hltSingleMuStartupLevel1Seed.L1SeedsLogicalExpression = 'L1_hltSingleMu3 OR L1_hltSingleMu5 OR L1_hltSingleMu7'

