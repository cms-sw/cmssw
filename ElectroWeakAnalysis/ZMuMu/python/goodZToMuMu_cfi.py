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


goodZToMuMuNotFiltered = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(False) ### not filtered, needed for AB and BB region study 

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



### exploring the 2.1 -- 2.4  eta region
### A: |eta|<2.1, B: 2.1<|eta|<2.4
zToMuMuABLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionABLoose,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)


### two muon with 2.1< eta < 2.4
zToMuMuBBLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionBBLoose,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)


zToMuMuAB = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionAB,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)

### two muon with 2.1< eta < 2.4
zToMuMuBB = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionBB,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)


goodZToMuMuABLoose = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuABLoose"),
    overlap = cms.InputTag("goodZToMuMuNotFiltered"),
    filter = cms.bool(True)
)


goodZToMuMuAB = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuAB"),
    overlap = cms.InputTag("goodZToMuMuNotFiltered"),
    filter = cms.bool(True)
)


goodZToMuMuAB1HLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuABLoose"),
    condition =cms.string("exactlyOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

goodZToMuMuBB2HLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("zToMuMuBBLoose"),
    condition =cms.string("bothMatched"),
    hltPath = cms.string("HLT_DoubleMu3"),
    filter = cms.bool(True) 
)



goodZToMuMuAB1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuAB"),
    condition =cms.string("exactlyOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)

goodZToMuMuBB2HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("zToMuMuBB"),
    condition =cms.string("bothMatched"),
    hltPath = cms.string("HLT_DoubleMu3"),
    filter = cms.bool(True) 
)










