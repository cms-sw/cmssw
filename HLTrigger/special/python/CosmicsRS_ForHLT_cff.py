import FWCore.ParameterSet.Config as cms

# roads
from RecoTracker.RoadMapMakerESProducer.RoadMapMakerESProducerP5_cff import *
import copy
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cfi import *
roadSearchSeedsP5 = copy.deepcopy(roadSearchSeeds)
import copy
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cfi import *
#remove cut on number of clusters
#replace roadSearchSeedsP5.MaxNumberOfCosmicClusters = 0
#include "RecoTracker/RoadSearchCloudMaker/data/RoadSearchCloudsP5.cff"
# RoadSearchCloudMaker
roadSearchCloudsP5 = copy.deepcopy(roadSearchClouds)
import copy
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cfi import *
rsTrackCandidatesP5 = copy.deepcopy(rsTrackCandidates)
import copy
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cfi import *
rsWithMaterialTracksP5 = copy.deepcopy(rsWithMaterialTracks)
hltTrackerCosmicsSeedsFilterRS = cms.EDFilter("HLTCountNumberOfRoadSearchSeed",
    src = cms.InputTag("roadSearchSeedsP5"),
    MaxN = cms.int32(50),
    MinN = cms.int32(-1)
)

hltTrackerCosmicsTracksFilterRS = cms.EDFilter("HLTCountNumberOfTrack",
    src = cms.InputTag("rsWithMaterialTracksP5"),
    MaxN = cms.int32(1000),
    MinN = cms.int32(1)
)

hltTrackerCosmicsSeedsRS = cms.Sequence(roadSearchSeedsP5)
hltTrackerCosmicsTracksRS = cms.Sequence(roadSearchCloudsP5+rsTrackCandidatesP5+cms.SequencePlaceholder("offlineBeamSpot")+rsWithMaterialTracksP5)
roadSearchSeedsP5.Mode = 'STRAIGHT-LINE'
roadSearchSeedsP5.doClusterCheck = True
roadSearchSeedsP5.ClusterCollectionLabel = 'SiStripRawToClustersFacility'
roadSearchSeedsP5.RoadsLabel = 'P5'
roadSearchSeedsP5.InnerSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsP5.InnerSeedRecHitAccessUseRPhi = True
roadSearchSeedsP5.InnerSeedRecHitAccessUseStereo = True
roadSearchSeedsP5.OuterSeedRecHitAccessMode = 'STANDARD'
roadSearchSeedsP5.OuterSeedRecHitAccessUseRPhi = True
roadSearchSeedsP5.OuterSeedRecHitAccessUseStereo = True
roadSearchCloudsP5.RoadsLabel = 'P5'
roadSearchCloudsP5.SeedProducer = 'roadSearchSeedsP5'
roadSearchCloudsP5.UsePixelsinRS = False
roadSearchCloudsP5.UseRphiRecHits = True
roadSearchCloudsP5.UseStereoRecHits = True
roadSearchCloudsP5.scalefactorRoadSeedWindow = 150
roadSearchCloudsP5.MinimumHalfRoad = 3.3
roadSearchCloudsP5.RPhiRoadSize = 5.
roadSearchCloudsP5.MinimalFractionOfUsedLayersPerCloud = 0.3 ##standard: 0.5

roadSearchCloudsP5.MaximalFractionOfMissedLayersPerCloud = 0.8 ##standard: 0.3

roadSearchCloudsP5.MaximalFractionOfConsecutiveMissedLayersPerCloud = 0.35 ##standard:0.15

roadSearchCloudsP5.IncreaseMaxNumberOfConsecutiveMissedLayersPerCloud = 0
roadSearchCloudsP5.IncreaseMaxNumberOfMissedLayersPerCloud = 0
roadSearchCloudsP5.StraightLineNoBeamSpotCloud = True
rsTrackCandidatesP5.CloudProducer = 'roadSearchCloudsP5'
rsTrackCandidatesP5.MeasurementTrackerName = ''
rsTrackCandidatesP5.StraightLineNoBeamSpotCloud = True
rsTrackCandidatesP5.CosmicTrackMerging = True
rsTrackCandidatesP5.HitChi2Cut = 30.0
rsTrackCandidatesP5.NumHitCut = 4 ##CHANGE TO 5

rsTrackCandidatesP5.MinimumChunkLength = 2
rsTrackCandidatesP5.nFoundMin = 2
rsWithMaterialTracksP5.src = 'rsTrackCandidatesP5'
rsWithMaterialTracksP5.beamSpot = 'offlineBeamSpot'

