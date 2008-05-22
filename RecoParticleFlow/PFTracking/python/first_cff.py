# The following comments couldn't be translated into the new config version:

#FILTER

import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
#TRAJECTORY FILTER
firstCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
#TRAJECTORY BUILDER
firstCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
#TRACK CANDIDATES
firstTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
#TRACKS
firstWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
firstTriplets = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGenerator
        )
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

firstvtxFilt = cms.EDFilter("VertexFilter",
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(5),
    DistRhoFromVertex = cms.double(0.1),
    recTracks = cms.InputTag("firstWithMaterialTracks"),
    DistZFromVertex = cms.double(0.5),
    ChiCut = cms.double(100.0)
)

first = cms.Sequence(firstTriplets*firstTrackCandidates*firstWithMaterialTracks*firstvtxFilt)
firstCkfTrajectoryFilter.ComponentName = 'firstCkfTrajectoryFilter'
firstCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 5
firstCkfTrajectoryBuilder.ComponentName = 'firstCkfTrajectoryBuilder'
firstCkfTrajectoryBuilder.trajectoryFilterName = 'firstCkfTrajectoryFilter'
firstTrackCandidates.SeedProducer = 'firstTriplets'
firstTrackCandidates.TrajectoryBuilder = 'firstCkfTrajectoryBuilder'
firstWithMaterialTracks.src = 'firstTrackCandidates'

