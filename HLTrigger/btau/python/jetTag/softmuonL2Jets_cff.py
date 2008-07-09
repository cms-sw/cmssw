import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBSoftmuon1jetL2filter = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBSoftmuon2jetL2filter = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBSoftmuon3jetL2filter = copy.deepcopy(hlt1CaloJet)
import copy
from HLTrigger.HLTfilters.hlt1CaloJet_cfi import *
hltBSoftmuon4jetL2filter = copy.deepcopy(hlt1CaloJet)
hltBSoftmuonHTL2filter = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("htMet"),
    MinN = cms.int32(1),
    Min = cms.double(370.0)
)

hltBSoftmuon1jetL2filter.MinN = 1
hltBSoftmuon1jetL2filter.MinPt = 20.0
hltBSoftmuon2jetL2filter.MinN = 2
hltBSoftmuon2jetL2filter.MinPt = 120.0
hltBSoftmuon3jetL2filter.MinN = 3
hltBSoftmuon3jetL2filter.MinPt = 70.0
hltBSoftmuon4jetL2filter.MinN = 4
hltBSoftmuon4jetL2filter.MinPt = 40.0

