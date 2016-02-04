import FWCore.ParameterSet.Config as cms
import copy
from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *



zToMuMuOneTrackerMuonLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionLoose,
    src = cms.InputTag("dimuonsOneTrackerMuon"),
    filter = cms.bool(True)
)


zToMuMuOneTrackerMuon = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneTrackerMuon"),
    filter = cms.bool(True)
)



## attention to the overlap... should be done with tight zmumu
goodZToMuMuOneTrackerMuonLoose = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuOneTrackerMuonLoose"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

goodZToMuMuOneTrackerMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuOneTrackerMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)





#ZMuTrkMuon:requiring that the GlobalMuon has HLT match
goodZToMuMuOneTrackerMuonFirstHLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrackerMuonLoose"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


goodZToMuMuOneTrackerMuonFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneTrackerMuon"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)




