import FWCore.ParameterSet.Config as cms
import copy
from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

zToMuGlobalMuOneTrack = cms.EDFilter(
    "CandViewRefSelector",
    cut = cms.string("daughter(0).isGlobalMuon = 1"),
    ### added UserData
    src = cms.InputTag("userDataDimuonsOneTrack"),
    ###src = cms.InputTag("dimuonsOneTrack"),
    filter = cms.bool(True)
)

zToMuMuOneTrackLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionLoose,
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    filter = cms.bool(True)
)


zToMuMuOneTrack = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("zToMuGlobalMuOneTrack"),
    filter = cms.bool(True)
)


## attention to the overlap... should be done with tight zmumu

goodZToMuMuOneTrackLoose = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrackLoose"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

goodZToMuMuOneTrack = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",
    src = cms.InputTag("zToMuMuOneTrack"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)



#goodZToMuMuOneTrack = copy.deepcopy(goodZTight)
#goodZToMuMuOneTrack.src = cms.InputTag("goodZToMuMuOneTrackLoose")


#ZMuTk:requiring that the GlobalMuon 'First' has HLT match
goodZToMuMuOneTrackFirstHLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrackLoose"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


#goodZToMuMuOneTrackFirstHLT = copy.deepcopy(goodZTight)
#goodZToMuMuOneTrackFirstHLT.src = cms.InputTag("goodZToMuMuOneTrackFirstHLTLoose")

goodZToMuMuOneTrackFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrack"),
    condition =cms.string("firstMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)
