import FWCore.ParameterSet.Config as cms

# Iterative steps
from RecoTracker.IterativeTracking.iterativeTk_cff import *


# RS
from RecoTracker.RoadSearchSeedFinder.RoadSearchSeeds_cff import *
from RecoTracker.RoadSearchCloudMaker.RoadSearchClouds_cff import *
from RecoTracker.RoadSearchTrackCandidateMaker.RoadSearchTrackCandidates_cff import *
from RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff import *

ckftracks = cms.Sequence(iterTracking*trackCollectionMerging)

rstracks = cms.Sequence(roadSearchSeeds*
                        roadSearchClouds*rsTrackCandidates*
                        rsWithMaterialTracks)



