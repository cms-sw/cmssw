import FWCore.ParameterSet.Config as cms

import copy

from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

goodZToMuMuLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionLoose,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)





#ZMuMu: requiring at least  1 HLT trigger match (for the shape)
goodZToMuMuAtLeast1HLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuLoose"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)




#ZMuMu: requiring  2 HLT trigger match
goodZToMuMu2HLTLoose = copy.deepcopy(goodZToMuMuAtLeast1HLTLoose)
goodZToMuMu2HLTLoose.condition =cms.string("bothMatched")

goodZToMuMu2HLT = copy.deepcopy(goodZToMuMuAtLeast1HLT)
goodZToMuMu2HLT.condition =cms.string("bothMatched")


#ZMuMu: requiring 1 HLT trigger match
goodZToMuMu1HLTLoose = copy.deepcopy(goodZToMuMuAtLeast1HLTLoose)
goodZToMuMu1HLTLoose.condition =cms.string("exactlyOneMatched")

goodZToMuMu1HLT = copy.deepcopy(goodZToMuMuAtLeast1HLT)
goodZToMuMu1HLT.condition =cms.string("exactlyOneMatched")

