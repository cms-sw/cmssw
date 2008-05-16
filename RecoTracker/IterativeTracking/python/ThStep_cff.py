import FWCore.ParameterSet.Config as cms

import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
thPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
thStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi import *
#SEEDS
thPLSeeds = copy.deepcopy(globalMixedSeeds)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
thMeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#TRAJECTORY FILTER
thCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
thCkfTrajectoryBuilder = copy.deepcopy(GroupedCkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
thTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
thWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
from RecoTracker.IterativeTracking.ThVxFilter_cff import *
#HIT REMOVAL
thClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("secClusters"),
    trajectories = cms.InputTag("secStep"),
    pixelClusters = cms.InputTag("secClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("secClusters")
)

#SEEDING LAYERS
thlayerpairs = cms.ESProducer("MixedLayerPairsESProducer",
    ComponentName = cms.string('ThLayerPairs'),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix1+FPix2_pos', 
        'BPix1+FPix2_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'BPix2+FPix2_pos', 
        'BPix2+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg', 
        'FPix2_pos+TEC1_pos', 
        'FPix2_pos+TEC2_pos', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'FPix2_neg+TEC1_neg', 
        'FPix2_neg+TEC2_neg', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('thPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('thPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

thirdStep = cms.Sequence(thClusters*thPixelRecHits*thStripRecHits*thPLSeeds*thTrackCandidates*thWithMaterialTracks*thStep)
thPixelRecHits.src = 'thClusters'
thStripRecHits.ClusterProducer = 'thClusters'
thPLSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'ThLayerPairs'
thPLSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.3
thPLSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 22.7
thMeasurementTracker.ComponentName = 'thMeasurementTracker'
thMeasurementTracker.pixelClusterProducer = 'thClusters'
thMeasurementTracker.stripClusterProducer = 'thClusters'
thCkfTrajectoryFilter.ComponentName = 'thCkfTrajectoryFilter'
thCkfTrajectoryFilter.filterPset.maxLostHits = 0
thCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
thCkfTrajectoryFilter.filterPset.minPt = 0.3
thCkfTrajectoryBuilder.ComponentName = 'thCkfTrajectoryBuilder'
thCkfTrajectoryBuilder.MeasurementTrackerName = 'thMeasurementTracker'
thCkfTrajectoryBuilder.trajectoryFilterName = 'thCkfTrajectoryFilter'
thTrackCandidates.SeedProducer = 'thPLSeeds'
thTrackCandidates.TrajectoryBuilder = 'thCkfTrajectoryBuilder'
thTrackCandidates.doSeedingRegionRebuilding = True
thTrackCandidates.useHitsSplitting = True
thWithMaterialTracks.src = 'thTrackCandidates'
thWithMaterialTracks.clusterRemovalInfo = 'thClusters'

