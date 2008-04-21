import FWCore.ParameterSet.Config as cms

from RecoTracker.SpecialSeedGenerators.CosmicSeedP5Pairs_cff import *
from RecoTracker.SingleTrackPattern.CosmicTrackFinderP5_cff import *
hltTrackerCosmicsSeedsFilterCoTF = cms.EDFilter("HLTCountNumberOfTrajectorySeed",
    src = cms.InputTag("cosmicseedfinderP5"),
    MaxN = cms.int32(50),
    MinN = cms.int32(-1)
)

hltTrackerCosmicsTracksFilterCoTF = cms.EDFilter("HLTCountNumberOfTrack",
    src = cms.InputTag("cosmictrackfinderP5"),
    MaxN = cms.int32(1000),
    MinN = cms.int32(1)
)

hltTrackerCosmicsSeedsCoTF = cms.Sequence(cosmicseedfinderP5)
hltTrackerCosmicsTracksCoTF = cms.Sequence(cosmictrackfinderP5)
cosmicseedfinderP5.ClusterCollectionLabel = 'SiStripRawToClustersFacility'

