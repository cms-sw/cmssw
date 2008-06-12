import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.blockHLT_cff import *

from RecoMuon.GlobalTrackingTools.MuonTrackingRegionCommon_cff import *
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
from RecoMuon.TrackerSeedGenerator.TrackerSeedCleaner_cff import *
# include  "RecoMuon/TrackerSeedGenerator/data/TSGs.cff"
hltL3TrajectorySeed = cms.EDFilter("FastTSGFromL2Muon",
    # ServiceParameters
    MuonServiceProxy,
    # The collection of Sim Tracks
    SimTrackCollectionLabel = cms.InputTag("famosSimHits"),
    # The STA muons for which seeds are looked for in the tracker
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    #using TrackerSeedCleanerCommon
    MuonTrackingRegionBuilder = cms.PSet(
        block_hltL3TrajectorySeed
    ),
    # Keep tracks with pT > 1 GeV 
    PtCut = cms.double(1.0),
    # The Tracks from which seeds are looked for
    SeedCollectionLabels = cms.VInputTag(cms.InputTag("pixelTripletSeeds","PixelTriplet"), cms.InputTag("globalPixelSeeds","GlobalPixel"))
)


