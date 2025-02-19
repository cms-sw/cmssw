import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *
import copy

#### tight only....

#ZMuMu:at least one muon is not isolated 
nonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsGlobal"),
    filter = cms.bool(True) 
)


#ZMuMu:1 muon is not isolated 
oneNonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuOneNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("nonIsolatedZToMuMu"),
    filter = cms.bool(True) 
)




#ZMuMu: 2 muons are not isolated 
twoNonIsolatedZToMuMu = cms.EDFilter(
    "ZToMuMuTwoNonIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("nonIsolatedZToMuMu"),
    filter = cms.bool(True) 
)



#ZMuMunotIso: requiring at least 1 trigger
nonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("nonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


#ZMuMuOnenotIso: requiring at least 1 trigger
oneNonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("oneNonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)



#ZMuMuTwonotIso: requiring at least 1 trigger
twoNonIsolatedZToMuMuAtLeast1HLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("twoNonIsolatedZToMuMu"),
    condition =cms.string("atLeastOneMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


