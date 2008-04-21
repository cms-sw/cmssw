import FWCore.ParameterSet.Config as cms

from FastSimulation.HighLevelTrigger.btau.pixelReco_cff import *
from FastSimulation.HighLevelTrigger.btau.lifetimeRegionalTracking_cff import *
from FastSimulation.HighLevelTrigger.btau.L3ForDisplacedMumuTrigger_cff import *
from FastSimulation.HighLevelTrigger.btau.L3ForMuMuk_cff import *
hltBLifetimeL1seeds.L1MuonCollectionTag = 'l1ParamMuons'
hltBSoftmuonNjetL1seeds.L1MuonCollectionTag = 'l1ParamMuons'
hltBSoftmuonHTL1seeds.L1MuonCollectionTag = 'l1ParamMuons'
JpsitoMumuL1Seed.L1MuonCollectionTag = 'l1ParamMuons'
MuMukL1Seed.L1MuonCollectionTag = 'l1ParamMuons'
hltBLifetimeL1seeds.L1GtObjectMapTag = 'gtDigis'
hltBSoftmuonNjetL1seeds.L1GtObjectMapTag = 'gtDigis'
hltBSoftmuonHTL1seeds.L1GtObjectMapTag = 'gtDigis'
JpsitoMumuL1Seed.L1GtObjectMapTag = 'gtDigis'
MuMukL1Seed.L1GtObjectMapTag = 'gtDigis'

