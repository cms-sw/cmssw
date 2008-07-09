import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBLifetime1jetL2filter = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBLifetime2jetL2filter = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBLifetime3jetL2filter = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBLifetime4jetL2filter = copy.deepcopy(hlt1CaloJet)
hltBLifetimeHTL2filter = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("htMet"),
    MinN = cms.int32(1),
    Min = cms.double(470.0)
)

hltBLifetime1jetL2filter.MinN = 1
hltBLifetime2jetL2filter.MinN = 2
hltBLifetime3jetL2filter.MinN = 3
hltBLifetime4jetL2filter.MinN = 4
hltBLifetime1jetL2filter.MinPt = 180.0
hltBLifetime2jetL2filter.MinPt = 120.0
hltBLifetime3jetL2filter.MinPt = 70.0
hltBLifetime4jetL2filter.MinPt = 40.0

