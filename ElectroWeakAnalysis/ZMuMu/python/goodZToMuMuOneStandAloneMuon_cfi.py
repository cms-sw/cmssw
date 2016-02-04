import FWCore.ParameterSet.Config as cms
import copy
from ElectroWeakAnalysis.ZMuMu.zSelection_cfi import *

zToMuMuOneStandAloneMuonLoose = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelectionLoose,
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    filter = cms.bool(True)
)

zToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZToMuMuIsolatedIDSelector",
    zSelection,
    src = cms.InputTag("dimuonsOneStandAloneMuon"),
    filter = cms.bool(True)
)


goodZToMuMuOneStandAloneMuonLoose = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuOneStandAloneMuonLoose"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)

## attention to the overlap... should be done for both tight and loose cuts

goodZToMuMuOneStandAloneMuon = cms.EDFilter(
    "ZMuMuOverlapExclusionSelector",    
    src = cms.InputTag("zToMuMuOneStandAloneMuon"),
    overlap = cms.InputTag("goodZToMuMu"),
    filter = cms.bool(True)
)



#goodZToMuMuOneStandAloneMuon = copy.deepcopy(goodZTight)
#goodZToMuMuOneStandAloneMuon.src = cms.InputTag("goodZToMuMuOneStandAloneMuonLoose")

#ZMuSta:requiring that the GlobalMuon has HLT match
goodZToMuMuOneStandAloneMuonFirstHLTLoose = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuonLoose"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)


goodZToMuMuOneStandAloneMuonFirstHLT = cms.EDFilter(
    "ZHLTMatchFilter",
    src = cms.InputTag("goodZToMuMuOneStandAloneMuon"),
    condition =cms.string("globalisMatched"),
    hltPath = cms.string("HLT_Mu9"),
    filter = cms.bool(True) 
)




#goodZToMuMuOneStandAloneMuonFirstHLTTight = copy.deepcopy(goodZTight)
#goodZToMuMuOneStandAloneMuonFirstHLTTight.src = cms.InputTag("goodZToMuMuOneStandAloneMuonFirstHLT")
