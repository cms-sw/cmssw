import FWCore.ParameterSet.Config as cms
import copy

from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

# same charge dimuons....
dimuonsGlobalSameChargeLoose = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuons"),
    cut = cms.string('mass > 0 & daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1')
)

goodZToMuMuSameChargeLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionLoose,
    src = cms.InputTag("dimuonsGlobalSameChargeLoose"),
    filter = cms.bool(True) 
)

goodZToMuMuSameChargeLoose.cut=cms.string("charge!=0 & daughter(0).pt > 20 & daughter(1).pt > 20 & abs(daughter(0).eta)<2.1 & abs(daughter(1).eta)<2.1 & mass > 20")




goodZToMuMuSameCharge = copy.deepcopy(goodZTight)
goodZToMuMuSameCharge.src = cms.InputTag("goodZToMuMuSameChargeLoose")


goodZToMuMuSameChargeAtLeast1HLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuSameChargeLoose"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


goodZToMuMuSameChargeAtLeast1HLT = copy.deepcopy(goodZTight)
goodZToMuMuSameChargeAtLeast1HLT.src = cms.InputTag("goodZToMuMuSameChargeAtLeast1HLTLoose")


goodZToMuMuSameCharge2HLTLoose = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLTLoose)
goodZToMuMuSameCharge2HLTLoose.condition= cms.string("bothMatched")

goodZToMuMuSameCharge1HLT = copy.deepcopy(goodZTight)
goodZToMuMuSameCharge1HLT.src = cms.InputTag("goodZToMuMuSameCharge1HLTLoose")


goodZToMuMuSameCharge1HLTLoose = copy.deepcopy(goodZToMuMuSameChargeAtLeast1HLTLoose)
goodZToMuMuSameCharge1HLTLoose.condition= cms.string("exactlyOneMatched")


goodZToMuMuSameCharge2HLT = copy.deepcopy(goodZTight)
goodZToMuMuSameCharge2HLT.src = cms.InputTag("goodZToMuMuSameCharge2HLTLoose")
