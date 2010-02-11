import FWCore.ParameterSet.Config as cms


from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

goodZToMuMu = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)


#ZMuMu: requiring at least  1 HLT trigger match (for the shape)
goodZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

import copy
#ZMuMu: requiring  2 HLT trigger match
goodZToMuMu2HLT = copy.deepcopy(goodZToMuMuAtLeast1HLT)
goodZToMuMu2HLT.condition =cms.string("bothMatched")


#ZMuMu: requiring 1 HLT trigger match
goodZToMuMu1HLT = copy.deepcopy(goodZToMuMuAtLeast1HLT)
goodZToMuMu1HLT.condition =cms.string("exactlyOneMatched")

