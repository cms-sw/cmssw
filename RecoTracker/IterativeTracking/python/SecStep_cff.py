import FWCore.ParameterSet.Config as cms

import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
secPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
secStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
#SEEDING LAYERS
#PIXEL
seclayertriplets = copy.deepcopy(pixellayertriplets)
import copy
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi import *
#SEEDS
#TRIPLETS
secTriplets = copy.deepcopy(globalSeedsFromTripletsWithVertices)
import copy
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi import *
#TRIPLETS
secPlTriplets = copy.deepcopy(globalSeedsFromTripletsWithVertices)
import copy
from RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi import *
#COMBINED 
secCombSeeds = copy.deepcopy(globalCombinedSeeds)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
secMeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#TRAJECTORY FILTER
secCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
secCkfTrajectoryBuilder = copy.deepcopy(GroupedCkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
secTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
secWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
#FINAL TRACKS
from RecoTracker.IterativeTracking.SecVxFilter_cff import *
#HIT REMOVAL
secClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("firstfilter"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

#TIB LAYERS
secPLlayertriplets = cms.ESProducer("MixedLayerTripletsESProducer",
    ComponentName = cms.string('SecPlLayerTriplets'),
    layerList = cms.vstring('TIB1+TIB2+TIB3'),
    TIB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("secStripRecHits","matchedRecHit"),
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        rphiRecHits = cms.InputTag("secStripRecHits","rphiRecHit")
    )
)

secondStep = cms.Sequence(secClusters*secPixelRecHits*secStripRecHits*secTriplets*secPlTriplets*secCombSeeds*secTrackCandidates*secWithMaterialTracks*secStep)
secPixelRecHits.src = cms.InputTag("secClusters")
secStripRecHits.ClusterProducer = 'secClusters'
seclayertriplets.ComponentName = 'SecLayerTriplets'
seclayertriplets.BPix.HitProducer = 'secPixelRecHits'
seclayertriplets.FPix.HitProducer = 'secPixelRecHits'
secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 22.7
secTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecLayerTriplets'
secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.3
secPlTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 22.7
secPlTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecPlLayerTriplets'
secPlTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.3
secPlTriplets.OrderedHitsFactoryPSet.GeneratorPSet.extraHitRZtolerance = 12.0
secCombSeeds.PairCollection = 'secTriplets'
secCombSeeds.TripletCollection = 'secPlTriplets'
secMeasurementTracker.ComponentName = 'secMeasurementTracker'
secMeasurementTracker.pixelClusterProducer = 'secClusters'
secMeasurementTracker.stripClusterProducer = 'secClusters'
secCkfTrajectoryFilter.ComponentName = 'secCkfTrajectoryFilter'
secCkfTrajectoryFilter.filterPset.maxLostHits = 1
secCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
secCkfTrajectoryFilter.filterPset.minPt = 0.3
secCkfTrajectoryBuilder.ComponentName = 'secCkfTrajectoryBuilder'
secCkfTrajectoryBuilder.MeasurementTrackerName = 'secMeasurementTracker'
secCkfTrajectoryBuilder.trajectoryFilterName = 'secCkfTrajectoryFilter'
secTrackCandidates.SeedProducer = 'secCombSeeds'
#replace secTrackCandidates.SeedProducer= "secPlTriplets"
secTrackCandidates.TrajectoryBuilder = 'secCkfTrajectoryBuilder'
secTrackCandidates.doSeedingRegionRebuilding = True
secWithMaterialTracks.src = 'secTrackCandidates'
secWithMaterialTracks.clusterRemovalInfo = 'secClusters'

