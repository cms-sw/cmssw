import FWCore.ParameterSet.Config as cms
import copy

from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

# same charge dimuons....
dimuonsGlobalSameCharge = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuons"),
    cut = cms.string('mass > 20 & daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1')
)

goodZToMuMuSameCharge = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobalSameCharge"),
    filter = cms.bool(True) 
)

goodZToMuMuSameCharge.cut=cms.string("charge!=0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 20")


goodZToMuMuSameChargeAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuSameCharge"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)



goodZToMuMuSameCharge2HLT = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLT)
goodZToMuMuSameCharge2HLT.condition= cms.string("bothMatched")


goodZToMuMuSameCharge1HLT = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLT)
goodZToMuMuSameCharge1HLT.condition= cms.string("exactlyOneMatched")



