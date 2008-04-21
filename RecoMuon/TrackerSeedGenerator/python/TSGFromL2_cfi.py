import FWCore.ParameterSet.Config as cms

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
from RecoMuon.TrackerSeedGenerator.TrackerSeedCleaner_cff import *
from RecoMuon.TrackerSeedGenerator.TSGs_cff import *
hltL3TrajectorySeedFromL2 = cms.EDFilter("TSGFromL2Muon",
    MuonTrackingRegionCommon,
    #defines a bunch of PSet with name the name of the Seed generator
    # to redefine do
    # replace hltL3TrajectorySeedFromL2.TSGFromPixelPairs.TTRHBuilder = ""
    TSGsBlock,
    TrackerSeedCleanerCommon,
    #ServiceParameters
    MuonServiceProxy,
    tkSeedGenerator = cms.string('TSGFromCombinedHits'),
    UseTFileService = cms.untracked.bool(False),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    #this should not exist there !
    PtCut = cms.double(1.0)
)


