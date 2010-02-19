import FWCore.ParameterSet.Config as cms

import copy

from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

goodZToMuMuLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionLoose,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

goodZToMuMu = copy.deepcopy(goodZTight)
goodZToMuMu.src = cms.InputTag("goodZToMuMuLoose")



#ZMuMu: requiring at least  1 HLT trigger match (for the shape)
goodZToMuMuAtLeast1HLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuLoose"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

#ZMuMu: requiring at least  1 HLT trigger match (for the shape)
goodZToMuMuAtLeast1HLT = copy.deepcopy(goodZTight)
goodZToMuMuAtLeast1HLT.src = cms.InputTag("goodZToMuMuAtLeast1HLTLoose")


#ZMuMu: requiring  2 HLT trigger match
goodZToMuMu2HLTLoose = copy.deepcopy(goodZToMuMuAtLeast1HLTLoose)
goodZToMuMu2HLTLoose.condition =cms.string("bothMatched")

goodZToMuMu2HLT = copy.deepcopy(goodZTight)
goodZToMuMu2HLT.src = cms.InputTag("goodZToMuMu2HLTLoose")

#ZMuMu: requiring 1 HLT trigger match
goodZToMuMu1HLTLoose = copy.deepcopy(goodZToMuMuAtLeast1HLTLoose)
goodZToMuMu1HLTLoose.condition =cms.string("exactlyOneMatched")

goodZToMuMu1HLT = copy.deepcopy(goodZTight)
goodZToMuMu1HLT.src = cms.InputTag("goodZToMuMu1HLTLoose")
