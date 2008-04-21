import FWCore.ParameterSet.Config as cms

import copy
from RecoMuon.L2MuonSeedGenerator.L2MuonSeeds_cfi import *
hltL2MuonSeeds = copy.deepcopy(L2MuonSeeds)
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
from FastSimulation.Muons.L3Muons_cff import *
import copy
from FastSimulation.Muons.L3Muons_cfi import *
hltL3Muons = copy.deepcopy(L3Muons)
from FastSimulation.HighLevelTrigger.common.Vertexing_cff import *
hltL3MuonTracks = cms.Sequence(globalPixelGSTracking*hltL3Muons)
pixelTracksForMuons = cms.Sequence(pixelGSTracking)
DiMuonIsoLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
DiMuonNoIsoLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
JpsiMMLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
multiMuonNoIsoLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
SameSignMuLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
MuLevel1PathLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
SingleMuIsoLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
SingleMuNoIsoLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
SingleMuPrescale3Level1Seed.L1MuonCollectionTag = 'l1ParamMuons'
SingleMuPrescale5Level1Seed.L1MuonCollectionTag = 'l1ParamMuons'
SingleMuPrescale710Level1Seed.L1MuonCollectionTag = 'l1ParamMuons'
SingleMuPrescale77Level1Seed.L1MuonCollectionTag = 'l1ParamMuons'
UpsilonMMLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
ZMMLevel1Seed.L1MuonCollectionTag = 'l1ParamMuons'
DiMuonIsoLevel1Seed.L1GtObjectMapTag = 'gtDigis'
DiMuonNoIsoLevel1Seed.L1GtObjectMapTag = 'gtDigis'
JpsiMMLevel1Seed.L1GtObjectMapTag = 'gtDigis'
multiMuonNoIsoLevel1Seed.L1GtObjectMapTag = 'gtDigis'
SameSignMuLevel1Seed.L1GtObjectMapTag = 'gtDigis'
MuLevel1PathLevel1Seed.L1GtObjectMapTag = 'gtDigis'
SingleMuIsoLevel1Seed.L1GtObjectMapTag = 'gtDigis'
SingleMuNoIsoLevel1Seed.L1GtObjectMapTag = 'gtDigis'
SingleMuPrescale3Level1Seed.L1GtObjectMapTag = 'gtDigis'
SingleMuPrescale5Level1Seed.L1GtObjectMapTag = 'gtDigis'
SingleMuPrescale710Level1Seed.L1GtObjectMapTag = 'gtDigis'
SingleMuPrescale77Level1Seed.L1GtObjectMapTag = 'gtDigis'
UpsilonMMLevel1Seed.L1GtObjectMapTag = 'gtDigis'
ZMMLevel1Seed.L1GtObjectMapTag = 'gtDigis'
hltL2MuonSeeds.GMTReadoutCollection = 'l1ParamMuons'
hltL2MuonSeeds.InputObjects = 'l1ParamMuons'
hltL3Muons.MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx")
hltL3Muons.L3TrajBuilderParameters.TrackerTrajectories = 'globalPixelGSWithMaterialTracks'
hltL3Muons.L3TrajBuilderParameters.StateOnTrackerBoundOutPropagator = 'SmartPropagator'

