import FWCore.ParameterSet.Config as cms
import copy

from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

# same charge dimuons....
dimuonsGlobalSameCharge = cms.EDFilter(
    "CandViewRefSelector",
    ### added UserData
    src = cms.InputTag("userDataDimuons"),
    ##src = cms.InputTag("dimuons"),
    cut = cms.string('charge!=0 & mass > 0 & daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1')
    )


goodZToMuMuSameChargeLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionLoose,
    src = cms.InputTag("dimuonsGlobalSameCharge"),
    filter = cms.bool(True) 
)

goodZToMuMuSameChargeLoose.cut=cms.string("charge!=0 & daughter(0).pt > 10 & daughter(1).pt > 10 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 ")

goodZToMuMuSameCharge = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobalSameCharge"),
    filter = cms.bool(True) 
)

goodZToMuMuSameCharge.cut=cms.string("charge!=0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 ")





goodZToMuMuSameChargeAtLeast1HLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuSameChargeLoose"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

goodZToMuMuSameChargeAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuSameCharge"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)





goodZToMuMuSameCharge2HLTLoose = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLTLoose)
goodZToMuMuSameCharge2HLTLoose.condition= cms.string("bothMatched")

goodZToMuMuSameCharge1HLT = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLT)
goodZToMuMuSameCharge1HLT.condition= cms.string("bothMatched")

goodZToMuMuSameCharge1HLTLoose = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLTLoose)
goodZToMuMuSameCharge1HLTLoose.condition= cms.string("exactlyOneMatched")


goodZToMuMuSameCharge2HLT = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLT)
goodZToMuMuSameCharge2HLT.condition= cms.string("exactlyOneMatched")
