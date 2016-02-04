import FWCore.ParameterSet.Config as cms

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
from RecoMuon.TrackerSeedGenerator.TrackerSeedCleaner_cff import *
from RecoMuon.TrackerSeedGenerator.TSGs_cff import *

hltL3TrajectorySeedFromL2 = cms.EDProducer("TSGFromL2Muon",
    # ServiceParameters
    MuonServiceProxy,
    # MuonTrackingRegionBuilder and  TrackerSeedCleaner should be empty for TSGForRoadSearchOI
    # MuonTrackingRegionBuilder should be empty for TSGFromPropagation
    #MuonTrackingRegionCommon,
    #TrackerSeedCleanerCommon,
    MuonTrackingRegionBuilder = cms.PSet(),
    TrackerSeedCleaner = cms.PSet(),
    TkSeedGenerator = TSGsBlock.TSGForRoadSearchOI,
    
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    #this should not exist there !
    PtCut = cms.double(1.0),
    PCut = cms.double(2.5)            
)



