# The following comments couldn't be translated into the new config version:

#FILTER

import FWCore.ParameterSet.Config as cms

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
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            ComponentName = cms.string('PixelTripletHLTGenerator'),
            extraHitRPhitolerance = cms.double(0.06),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.06)
        )
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originRadius = cms.double(0.2),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originZPos = cms.double(0.0)
        )
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

