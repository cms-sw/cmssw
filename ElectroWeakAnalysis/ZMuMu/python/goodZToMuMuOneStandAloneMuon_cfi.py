import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

zToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    filter = cms.bool(True)
)


goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuOneStandAloneMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

#ZMuSta:requiring that the GlobalMuon has HLT match
goodZToMuMuOneStandAloneMuonFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)
