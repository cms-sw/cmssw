# The following comments couldn't be translated into the new config version:

#FILTER

import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#TRAJECTORY FILTER
firstCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
firstCkfTrajectoryBuilder = copy.deepcopy(GroupedCkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
firstTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
firstWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
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

